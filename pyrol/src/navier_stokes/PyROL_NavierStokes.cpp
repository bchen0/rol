// @HEADER
// *****************************************************************************
//               Rapid Optimization Library (ROL) Package
//
// Copyright 2014 NTESS and the ROL contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Teuchos_Comm.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_MultiVector.hpp"

#include "ROL_ParameterList.hpp"
#include "ROL_PartitionedVector.hpp"
#include "ROL_ReducedDynamicObjective.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_Stream.hpp"

#include "dynconstraint.hpp"
#include "ltiobjective.hpp"
#include "meshreader.hpp"
#include "pdeobjective.hpp"
#include "pdevector.hpp"

#include "dynpde_navier-stokes.hpp"
#include "initial_condition.hpp"
#include "obj_navier-stokes.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace py = pybind11;

namespace {

using RealT = double;

bool isAbsolutePath(const std::string &path) {
  return !path.empty() && path[0] == '/';
}

std::string currentWorkingDirectory() {
  char buffer[PATH_MAX];
  if (::getcwd(buffer, sizeof(buffer)) == nullptr) {
    throw std::runtime_error(std::string("getcwd failed: ") + std::strerror(errno));
  }
  return std::string(buffer);
}

std::string dirnameOf(const std::string &path) {
  const std::string::size_type pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return currentWorkingDirectory();
  }
  if (pos == 0) {
    return "/";
  }
  return path.substr(0, pos);
}

std::string joinPath(const std::string &lhs, const std::string &rhs) {
  if (lhs.empty() || lhs == ".") {
    return rhs;
  }
  if (rhs.empty()) {
    return lhs;
  }
  if (isAbsolutePath(rhs)) {
    return rhs;
  }
  if (lhs[lhs.size() - 1] == '/') {
    return lhs + rhs;
  }
  return lhs + "/" + rhs;
}

std::string absolutePath(const std::string &path) {
  if (path.empty()) {
    throw std::invalid_argument("path must not be empty");
  }
  return isAbsolutePath(path) ? path : joinPath(currentWorkingDirectory(), path);
}

bool directoryExists(const std::string &path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

void ensureDirectory(const std::string &path) {
  if (directoryExists(path)) {
    return;
  }
  if (::mkdir(path.c_str(), 0777) != 0 && !directoryExists(path)) {
    throw std::runtime_error("could not create cache_dir '" + path
                             + "': " + std::strerror(errno));
  }
}

class ScopedWorkingDirectory {
public:
  explicit ScopedWorkingDirectory(const std::string &path)
    : previous_(currentWorkingDirectory()), active_(true) {
    if (::chdir(path.c_str()) != 0) {
      throw std::runtime_error("could not enter directory '" + path
                               + "': " + std::strerror(errno));
    }
  }

  ~ScopedWorkingDirectory() {
    if (active_) {
      ::chdir(previous_.c_str());
    }
  }

  ScopedWorkingDirectory(const ScopedWorkingDirectory &) = delete;
  ScopedWorkingDirectory &operator=(const ScopedWorkingDirectory &) = delete;

private:
  std::string previous_;
  bool active_;
};

void ensureTpetraInitialized() {
  static std::once_flag once;
  std::call_once(once, []() {
    if (!Tpetra::isInitialized()) {
      int argc = 1;
      char arg0[] = "pyrol._navier_stokes";
      char *argv[] = {arg0, nullptr};
      char **argvp = argv;
      Tpetra::initialize(&argc, &argvp);
      std::atexit([]() {
        if (Tpetra::isInitialized()) {
          Tpetra::finalize();
        }
      });
    }
  });
  if (!Tpetra::isInitialized()) {
    throw std::runtime_error("Tpetra failed to initialize");
  }
}

template<class Real>
void computeInitialConditionForDuration(
    const ROL::Ptr<ROL::Vector<Real>>       &u0,
    const ROL::Ptr<ROL::Vector<Real>>       &ck,
    const ROL::Ptr<ROL::Vector<Real>>       &uo,
    const ROL::Ptr<ROL::Vector<Real>>       &un,
    const ROL::Ptr<ROL::Vector<Real>>       &zk,
    const ROL::Ptr<DynConstraint<Real>>     &con,
    const Real                               dt,
    const Real                               spinupTime) {
  if (spinupTime <= static_cast<Real>(0)) {
    return;
  }

  const int nt = static_cast<int>(spinupTime / dt);
  if (nt <= 1) {
    return;
  }

  std::vector<ROL::TimeStamp<Real>> ts(nt);
  for (int k = 0; k < nt; ++k) {
    ts.at(k).t.resize(2);
    ts.at(k).t.at(0) = k * dt;
    ts.at(k).t.at(1) = (k + 1) * dt;
  }

  zk->zero();
  uo->set(*u0);
  un->zero();
  for (int k = 1; k < nt; ++k) {
    con->solve(*ck, *uo, *un, *zk, ts[k]);
    uo->set(*un);
  }
  u0->set(*uo);
}

class EvaluationGuard {
public:
  explicit EvaluationGuard(std::atomic<bool> &busy) : busy_(busy) {
    bool expected = false;
    if (!busy_.compare_exchange_strong(expected, true)) {
      throw std::runtime_error("NavierStokesObjective is already evaluating");
    }
  }

  ~EvaluationGuard() {
    busy_.store(false);
  }

  EvaluationGuard(const EvaluationGuard &) = delete;
  EvaluationGuard &operator=(const EvaluationGuard &) = delete;

private:
  std::atomic<bool> &busy_;
};

class NavierStokesObjective {
public:
  NavierStokesObjective(const std::string &xmlPath,
                        const std::string &cacheDir,
                        const RealT spinupTime)
    : xmlPath_(absolutePath(xmlPath)),
      xmlDir_(dirnameOf(xmlPath_)),
      cacheDir_(cacheDir.empty() ? xmlDir_ : absolutePath(cacheDir)),
      spinupTime_(spinupTime < static_cast<RealT>(0) ? static_cast<RealT>(80) : spinupTime),
      busy_(false) {
    ensureTpetraInitialized();

    comm_ = Tpetra::getDefaultComm();
    rank_ = comm_->getRank();
    commSize_ = comm_->getSize();

    if (rank_ == 0) {
      ensureDirectory(cacheDir_);
    }
    comm_->barrier();

    parlist_ = ROL::getParametersFromXmlFile(xmlPath_);
    patchParameterList();

    nt_ = parlist_->sublist("Time Discretization").get("Number of Time Steps", 100);
    if (nt_ < 2) {
      throw std::invalid_argument("Number of Time Steps must be at least 2");
    }
    const RealT T = parlist_->sublist("Time Discretization").get("End Time", 1.0);
    dt_ = T / static_cast<RealT>(nt_);

    const bool useParametricControl =
      parlist_->sublist("Problem").get("Use Parametric Control", false);
    if (!useParametricControl) {
      throw std::invalid_argument(
        "NavierStokesObjective currently supports only parametric control; "
        "set Problem/Use Parametric Control to true");
    }

    buildObjective();
  }

  int numSteps() const {
    return nt_;
  }

  int numControls() const {
    return nt_;
  }

  int commRank() const {
    return rank_;
  }

  int commSize() const {
    return commSize_;
  }

  RealT value(const py::array_t<RealT, py::array::c_style | py::array::forcecast> &z,
              const RealT tol) {
    const std::vector<RealT> values = copyInput(z, "z");
    RealT objectiveValue = 0;
    {
      EvaluationGuard guard(busy_);
      copyToPartitionedVector(*z_, values);
      RealT localTol = tol;
      {
        py::gil_scoped_release release;
        objective_->update(*z_, true, -1);
        objectiveValue = objective_->value(*z_, localTol);
      }
    }
    return objectiveValue;
  }

  py::array_t<RealT> gradient(
      const py::array_t<RealT, py::array::c_style | py::array::forcecast> &z,
      const RealT tol) {
    const std::vector<RealT> values = copyInput(z, "z");
    std::vector<RealT> result;
    {
      EvaluationGuard guard(busy_);
      copyToPartitionedVector(*z_, values);
      RealT localTol = tol;
      {
        py::gil_scoped_release release;
        objective_->update(*z_, true, -1);
        g_->zero();
        objective_->gradient(*g_, *z_, localTol);
        result = copyFromPartitionedVector(*g_);
      }
    }
    return vectorToArray(result);
  }

  py::array_t<RealT> hessVec(
      const py::array_t<RealT, py::array::c_style | py::array::forcecast> &v,
      const py::array_t<RealT, py::array::c_style | py::array::forcecast> &z,
      const RealT tol) {
    const std::vector<RealT> direction = copyInput(v, "v");
    const std::vector<RealT> values = copyInput(z, "z");
    std::vector<RealT> result;
    {
      EvaluationGuard guard(busy_);
      copyToPartitionedVector(*z_, values);
      copyToPartitionedVector(*v_, direction);
      RealT localTol = tol;
      {
        py::gil_scoped_release release;
        objective_->update(*z_, true, -1);
        hv_->zero();
        objective_->hessVec(*hv_, *v_, *z_, localTol);
        result = copyFromPartitionedVector(*hv_);
      }
    }
    return vectorToArray(result);
  }

  py::tuple valueAndGradient(
      const py::array_t<RealT, py::array::c_style | py::array::forcecast> &z,
      const RealT tol) {
    const std::vector<RealT> values = copyInput(z, "z");
    RealT objectiveValue = 0;
    std::vector<RealT> gradientValues;
    {
      EvaluationGuard guard(busy_);
      copyToPartitionedVector(*z_, values);
      RealT localTol = tol;
      {
        py::gil_scoped_release release;
        objective_->update(*z_, true, -1);
        objectiveValue = objective_->value(*z_, localTol);
        g_->zero();
        objective_->gradient(*g_, *z_, localTol);
        gradientValues = copyFromPartitionedVector(*g_);
      }
    }
    return py::make_tuple(objectiveValue, vectorToArray(gradientValues));
  }

private:
  void patchParameterList() {
    ROL::ParameterList &meshList = parlist_->sublist("Mesh");
    const std::string meshFile = meshList.get("File Name", "mesh.txt");
    if (!isAbsolutePath(meshFile)) {
      meshList.set("File Name", joinPath(xmlDir_, meshFile));
    }

    parlist_->sublist("General").set("Print Verbosity", 0);
    parlist_->sublist("Problem").set("Check Derivatives", false);
    parlist_->sublist("Problem").set("Print Uncontrolled State", false);
    parlist_->sublist("Dynamic Constraint").sublist("Solve")
      .set("Output Iteration History", false);
    parlist_->sublist("SimOpt").sublist("Solve")
      .set("Output Iteration History", false);

    ROL::ParameterList &rpl = parlist_->sublist("Reduced Dynamic Objective");
    rpl.set("State Domain Seed", 12321 * (rank_ + 1));
    rpl.set("State Range Seed", 32123 * (rank_ + 1));
    rpl.set("Adjoint Domain Seed", 23432 * (rank_ + 1));
    rpl.set("Adjoint Range Seed", 43234 * (rank_ + 1));
    rpl.set("State Sensitivity Domain Seed", 34543 * (rank_ + 1));
    rpl.set("State Sensitivity Range Seed", 54345 * (rank_ + 1));
  }

  void buildObjective() {
    const int partitionType =
      parlist_->sublist("Geometry").get("Partition type", 1);
    const int meshPartitions =
      (partitionType == 3 && commSize_ > 1) ? commSize_ : 0;
    outStream_ = ROL::makeStreamPtr(std::cout, false);

    meshMgr_ = ROL::makePtr<MeshReader<RealT>>(*parlist_, meshPartitions);
    pde_ = ROL::makePtr<DynamicPDE_NavierStokes<RealT>>(*parlist_);

    dynCon_ = ROL::makePtr<DynConstraint<RealT>>(pde_, meshMgr_, comm_, *parlist_, *outStream_);
    assembler_ = dynCon_->getAssembler();
    dynCon_->setSolveParameters(*parlist_);
    dynCon_->getAssembler()->printMeshData(*outStream_);

    u0Ptr_ = assembler_->createStateVector();
    ROL::Ptr<Tpetra::MultiVector<>> uoPtr = assembler_->createStateVector();
    ROL::Ptr<Tpetra::MultiVector<>> unPtr = assembler_->createStateVector();
    ROL::Ptr<Tpetra::MultiVector<>> ckPtr = assembler_->createResidualVector();

    u0_ = ROL::makePtr<PDE_PrimalSimVector<RealT>>(u0Ptr_, pde_, *assembler_, *parlist_);
    uo_ = ROL::makePtr<PDE_PrimalSimVector<RealT>>(uoPtr, pde_, *assembler_, *parlist_);
    un_ = ROL::makePtr<PDE_PrimalSimVector<RealT>>(unPtr, pde_, *assembler_, *parlist_);
    ck_ = ROL::makePtr<PDE_DualSimVector<RealT>>(ckPtr, pde_, *assembler_, *parlist_);

    zk_ = ROL::makePtr<PDE_OptVector<RealT>>(
      ROL::makePtr<ROL::StdVector<RealT>>(1));
    z_ = ROL::PartitionedVector<RealT>::create(*zk_, nt_);
    g_ = ROL::dynamicPtrCast<ROL::PartitionedVector<RealT>>(z_->dual().clone());
    v_ = ROL::dynamicPtrCast<ROL::PartitionedVector<RealT>>(z_->clone());
    hv_ = ROL::dynamicPtrCast<ROL::PartitionedVector<RealT>>(z_->dual().clone());

    std::vector<ROL::Ptr<QoI<RealT>>> qoiVec(3, ROL::nullPtr);
    std::vector<ROL::Ptr<QoI<RealT>>> qoiTerminal(1, ROL::nullPtr);
    const RealT w1 = parlist_->sublist("Problem").get("State Cost", 1.0);
    const RealT w2 = parlist_->sublist("Problem").get("State Boundary Cost", 1.0);
    const RealT w3 = parlist_->sublist("Problem").get("Control Cost", 0.0);
    const RealT wT = parlist_->sublist("Problem").get("Final Time State Cost", 1.0);
    std::vector<RealT> weights = {w1, w2, w3};
    std::vector<RealT> terminalWeights = {wT};

    const std::string integratedObjective =
      parlist_->sublist("Problem").get("Integrated Objective Type", "Dissipation");
    const std::string finalObjective =
      parlist_->sublist("Problem").get("Final Time Objective Type", "Tracking");

    qoiVec[0] = ROL::makePtr<QoI_State_NavierStokes<RealT>>(
      integratedObjective,
      *parlist_,
      pde_->getVelocityFE(),
      pde_->getPressureFE(),
      pde_->getFieldHelper());
    qoiVec[1] = ROL::makePtr<QoI_DownStreamPower_NavierStokes<RealT>>(
      pde_->getVelocityFE(),
      pde_->getPressureFE(),
      pde_->getVelocityBdryFE(1),
      pde_->getBdryCellLocIds(1),
      pde_->getFieldHelper());
    qoiVec[2] = ROL::makePtr<QoI_RotationControl_NavierStokes<RealT>>();
    qoiTerminal[0] = ROL::makePtr<QoI_State_NavierStokes<RealT>>(
      finalObjective,
      *parlist_,
      pde_->getVelocityFE(),
      pde_->getPressureFE(),
      pde_->getFieldHelper());

    ROL::Ptr<ROL::Objective_SimOpt<RealT>> objK =
      ROL::makePtr<PDE_Objective<RealT>>(qoiVec, weights, assembler_);
    ROL::Ptr<ROL::Objective_SimOpt<RealT>> objT =
      ROL::makePtr<PDE_Objective<RealT>>(qoiTerminal, terminalWeights, assembler_);
    dynObj_ = ROL::makePtr<LTI_Objective<RealT>>(*parlist_, objK, objT);

    timeStamp_.resize(nt_);
    for (int k = 0; k < nt_; ++k) {
      timeStamp_.at(k).t.resize(2);
      timeStamp_.at(k).t.at(0) = k * dt_;
      timeStamp_.at(k).t.at(1) = (k + 1) * dt_;
    }

    initializeState();

    ROL::ParameterList &rpl = parlist_->sublist("Reduced Dynamic Objective");
    objective_ = ROL::makePtr<ROL::ReducedDynamicObjective<RealT>>(
      dynObj_, dynCon_, u0_, zk_, ck_, timeStamp_, rpl, outStream_);
  }

  void initializeState() {
    const RealT reynoldsNumber =
      parlist_->sublist("Problem").get("Reynolds Number", 200.0);
    std::ostringstream name;
    name << "initial_condition_Re" << static_cast<int>(reynoldsNumber) << ".txt";
    const std::string cacheFile = name.str();

    ScopedWorkingDirectory cwd(cacheDir_);
    std::ifstream infile(cacheFile.c_str());
    if (infile.good()) {
      infile.close();
      dynCon_->inputTpetraVector(u0Ptr_, cacheFile);
      return;
    }

    PotentialFlow<RealT> potentialFlow(
      pde_->getVelocityFE(),
      pde_->getPressureFE(),
      pde_->getCellNodes(),
      assembler_->getDofManager()->getCellDofs(),
      assembler_->getCellIds(),
      pde_->getFieldHelper(),
      *parlist_);
    potentialFlow.build(u0Ptr_);

    computeInitialConditionForDuration<RealT>(
      u0_, ck_, uo_, un_, zk_, dynCon_, dt_, spinupTime_);
    dynCon_->outputTpetraVector(u0Ptr_, cacheFile);
  }

  std::vector<RealT> copyInput(
      const py::array_t<RealT, py::array::c_style | py::array::forcecast> &array,
      const char *name) const {
    const py::buffer_info info = array.request();
    if (info.ndim != 1) {
      throw std::invalid_argument(std::string(name) + " must be a 1-D NumPy array");
    }
    if (info.shape[0] != nt_) {
      std::ostringstream message;
      message << name << " must have length " << nt_ << "; got " << info.shape[0];
      throw std::invalid_argument(message.str());
    }
    const RealT *data = static_cast<const RealT *>(info.ptr);
    return std::vector<RealT>(data, data + nt_);
  }

  void copyToPartitionedVector(ROL::PartitionedVector<RealT> &target,
                               const std::vector<RealT> &values) const {
    if (target.numVectors() != static_cast<ROL::PartitionedVector<RealT>::size_type>(nt_)) {
      throw std::runtime_error("internal ROL vector has an unexpected number of partitions");
    }
    for (int k = 0; k < nt_; ++k) {
      ROL::Ptr<PDE_OptVector<RealT>> zk =
        ROL::dynamicPtrCast<PDE_OptVector<RealT>>(target.get(k));
      if (zk == ROL::nullPtr || zk->getParameter() == ROL::nullPtr) {
        throw std::runtime_error("internal ROL vector is not a parametric control vector");
      }
      ROL::Ptr<std::vector<RealT>> parameter = zk->getParameter()->getVector();
      if (parameter->size() != 1) {
        throw std::runtime_error("expected exactly one scalar parameter per time step");
      }
      (*parameter)[0] = values[k];
    }
  }

  std::vector<RealT> copyFromPartitionedVector(ROL::PartitionedVector<RealT> &source) const {
    std::vector<RealT> values(nt_, 0);
    if (source.numVectors() != static_cast<ROL::PartitionedVector<RealT>::size_type>(nt_)) {
      throw std::runtime_error("internal ROL vector has an unexpected number of partitions");
    }
    for (int k = 0; k < nt_; ++k) {
      ROL::Ptr<PDE_OptVector<RealT>> zk =
        ROL::dynamicPtrCast<PDE_OptVector<RealT>>(source.get(k));
      if (zk == ROL::nullPtr || zk->getParameter() == ROL::nullPtr) {
        throw std::runtime_error("internal ROL vector is not a parametric control vector");
      }
      ROL::Ptr<std::vector<RealT>> parameter = zk->getParameter()->getVector();
      if (parameter->size() != 1) {
        throw std::runtime_error("expected exactly one scalar parameter per time step");
      }
      values[k] = (*parameter)[0];
    }
    return values;
  }

  py::array_t<RealT> vectorToArray(const std::vector<RealT> &values) const {
    py::array_t<RealT> array(values.size());
    py::buffer_info info = array.request();
    RealT *data = static_cast<RealT *>(info.ptr);
    std::copy(values.begin(), values.end(), data);
    return array;
  }

  std::string xmlPath_;
  std::string xmlDir_;
  std::string cacheDir_;
  RealT spinupTime_;
  int nt_ = 0;
  RealT dt_ = 0;
  int rank_ = 0;
  int commSize_ = 1;

  ROL::Ptr<const Teuchos::Comm<int>> comm_;
  ROL::Ptr<ROL::ParameterList> parlist_;
  ROL::Ptr<std::ostream> outStream_;
  ROL::Ptr<MeshManager<RealT>> meshMgr_;
  ROL::Ptr<DynamicPDE_NavierStokes<RealT>> pde_;
  ROL::Ptr<DynConstraint<RealT>> dynCon_;
  ROL::Ptr<Assembler<RealT>> assembler_;
  ROL::Ptr<LTI_Objective<RealT>> dynObj_;
  ROL::Ptr<ROL::ReducedDynamicObjective<RealT>> objective_;
  std::vector<ROL::TimeStamp<RealT>> timeStamp_;

  ROL::Ptr<Tpetra::MultiVector<>> u0Ptr_;
  ROL::Ptr<ROL::Vector<RealT>> u0_;
  ROL::Ptr<ROL::Vector<RealT>> uo_;
  ROL::Ptr<ROL::Vector<RealT>> un_;
  ROL::Ptr<ROL::Vector<RealT>> ck_;
  ROL::Ptr<ROL::Vector<RealT>> zk_;
  ROL::Ptr<ROL::PartitionedVector<RealT>> z_;
  ROL::Ptr<ROL::PartitionedVector<RealT>> g_;
  ROL::Ptr<ROL::PartitionedVector<RealT>> v_;
  ROL::Ptr<ROL::PartitionedVector<RealT>> hv_;

  std::atomic<bool> busy_;
};

} // namespace

PYBIND11_MODULE(_navier_stokes, m) {
  m.doc() = "PyROL dynamic Navier-Stokes reduced objective bindings";

  py::class_<NavierStokesObjective>(m, "_NavierStokesObjective")
    .def(py::init<const std::string &, const std::string &, RealT>(),
         py::arg("xml_path"),
         py::arg("cache_dir") = "",
         py::arg("spinup_time") = -1.0)
    .def_property_readonly("num_steps", &NavierStokesObjective::numSteps)
    .def_property_readonly("num_controls", &NavierStokesObjective::numControls)
    .def_property_readonly("comm_rank", &NavierStokesObjective::commRank)
    .def_property_readonly("comm_size", &NavierStokesObjective::commSize)
    .def("value", &NavierStokesObjective::value,
         py::arg("z"), py::arg("tol") = 1e-8)
    .def("gradient", &NavierStokesObjective::gradient,
         py::arg("z"), py::arg("tol") = 1e-8)
    .def("hess_vec", &NavierStokesObjective::hessVec,
         py::arg("v"), py::arg("z"), py::arg("tol") = 1e-8)
    .def("value_and_gradient", &NavierStokesObjective::valueAndGradient,
         py::arg("z"), py::arg("tol") = 1e-8);
}
