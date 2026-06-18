#ifndef PYROL_FLOW_OPT_COMMON_HPP
#define PYROL_FLOW_OPT_COMMON_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ROL_Objective.hpp"
#include "ROL_ParameterList.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_Stream.hpp"
#include "ROL_TpetraMultiVector.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_MultiVector.hpp"
#include "assembler.hpp"
#include "meshreader.hpp"
#include "pdeconstraint.hpp"
#include "pdeobjective.hpp"
#include "pdevector.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace pyrol_flow_opt {

using RealT = double;
using TpetraMV = Tpetra::MultiVector<RealT>;

inline void ensureTpetraInitialized() {
  static bool registeredFinalizer = false;
  if (!Tpetra::isInitialized()) {
    int argc = 0;
    char **argv = nullptr;
    Tpetra::initialize(&argc, &argv);
    if (!registeredFinalizer) {
      std::atexit([]() {
        if (Tpetra::isInitialized()) {
          Tpetra::finalize();
        }
      });
      registeredFinalizer = true;
    }
  }
}

inline bool isAbsolutePath(const std::string &path) {
  return !path.empty() && path[0] == '/';
}

inline std::string dirname(const std::string &path) {
  const std::size_t pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return ".";
  }
  if (pos == 0) {
    return "/";
  }
  return path.substr(0, pos);
}

inline std::string joinPath(const std::string &base, const std::string &rel) {
  if (rel.empty() || isAbsolutePath(rel)) {
    return rel;
  }
  if (base.empty() || base == ".") {
    return rel;
  }
  return base[base.size() - 1] == '/' ? base + rel : base + "/" + rel;
}

inline void absolutizeMeshPath(ROL::ParameterList &parlist,
                               const std::string &xmlDir) {
  if (parlist.isSublist("Mesh")) {
    ROL::ParameterList &mesh = parlist.sublist("Mesh");
    if (mesh.isParameter("File Name")) {
      const std::string meshFile = mesh.get<std::string>("File Name");
      if (!isAbsolutePath(meshFile)) {
        mesh.set("File Name", joinPath(xmlDir, meshFile));
      }
    }
  }
  if (!parlist.isSublist("Geometry")) {
    return;
  }
  ROL::ParameterList &geometry = parlist.sublist("Geometry");
  if (!geometry.isParameter("Mesh File")) {
    return;
  }
  const std::string meshFile = geometry.get<std::string>("Mesh File");
  if (!isAbsolutePath(meshFile)) {
    geometry.set("Mesh File", joinPath(xmlDir, meshFile));
  }
}

inline py::array_t<RealT> makeArray(const std::vector<RealT> &values) {
  py::array_t<RealT> out(values.size());
  py::buffer_info info = out.request();
  std::memcpy(info.ptr, values.data(), values.size() * sizeof(RealT));
  return out;
}

class FlowOptObjectiveBase {
public:
  explicit FlowOptObjectiveBase(const std::string &xmlPath)
      : xmlPath_(xmlPath), xmlDir_(dirname(xmlPath)), rank_(0), size_(1),
        useParamVar_(false), localFieldSize_(0), globalFieldSize_(0),
        parameterSize_(0) {
    ensureTpetraInitialized();
    comm_ = Tpetra::getDefaultComm();
    rank_ = comm_->getRank();
    size_ = comm_->getSize();

    parlist_ = ROL::makePtr<ROL::ParameterList>();
    Teuchos::updateParametersFromXmlFile(xmlPath_, parlist_.ptr());
    absolutizeMeshPath(*parlist_, xmlDir_);

    if (parlist_->isSublist("Problem")) {
      ROL::ParameterList &problem = parlist_->sublist("Problem");
      problem.set("Check derivatives", false);
      problem.set("Solve Optimization Problem", false);
      useParamVar_ =
          problem.get("Use Optimal Constant Velocity", true);
    }
    if (parlist_->isSublist("SimOpt")) {
      ROL::ParameterList &simopt = parlist_->sublist("SimOpt");
      simopt.sublist("Solve").set("Output Iteration History", false);
    }
  }

  virtual ~FlowOptObjectiveBase() = default;

  int commRank() const { return rank_; }
  int commSize() const { return size_; }
  std::size_t localFieldSize() const { return localFieldSize_; }
  std::size_t globalFieldSize() const { return globalFieldSize_; }
  std::size_t parameterSize() const { return parameterSize_; }
  std::size_t localSize() const { return localFieldSize_ + parameterSize_; }
  bool usesParameterControl() const { return useParamVar_; }

  RealT value(const py::array_t<RealT, py::array::c_style |
                                      py::array::forcecast> &z,
              RealT tol) {
    copyArrayToVector(z, *zp_);
    objective_->update(*zp_, true, -1);
    return objective_->value(*zp_, tol);
  }

  py::array_t<RealT>
  gradient(const py::array_t<RealT, py::array::c_style |
                                    py::array::forcecast> &z,
           RealT tol) {
    copyArrayToVector(z, *zp_);
    objective_->update(*zp_, true, -1);
    g_->zero();
    objective_->gradient(*g_, *zp_, tol);
    return copyVectorToArray(*g_);
  }

  py::array_t<RealT>
  hessVec(const py::array_t<RealT, py::array::c_style |
                                   py::array::forcecast> &v,
          const py::array_t<RealT, py::array::c_style |
                                    py::array::forcecast> &z,
          RealT tol) {
    copyArrayToVector(z, *zp_);
    copyArrayToVector(v, *v_);
    objective_->update(*zp_, true, -1);
    hv_->zero();
    objective_->hessVec(*hv_, *v_, *zp_, tol);
    return copyVectorToArray(*hv_);
  }

  py::tuple valueAndGradient(
      const py::array_t<RealT, py::array::c_style | py::array::forcecast> &z,
      RealT tol) {
    copyArrayToVector(z, *zp_);
    objective_->update(*zp_, true, -1);
    const RealT val = objective_->value(*zp_, tol);
    g_->zero();
    objective_->gradient(*g_, *zp_, tol);
    return py::make_tuple(val, copyVectorToArray(*g_));
  }

  RealT gradientDot(
      const py::array_t<RealT, py::array::c_style | py::array::forcecast> &z,
      const py::array_t<RealT, py::array::c_style | py::array::forcecast> &v,
      RealT tol) {
    copyArrayToVector(z, *zp_);
    copyArrayToVector(v, *v_);
    objective_->update(*zp_, true, -1);
    g_->zero();
    objective_->gradient(*g_, *zp_, tol);
    return g_->dot(*v_);
  }

  RealT hessVecDot(
      const py::array_t<RealT, py::array::c_style | py::array::forcecast> &v,
      const py::array_t<RealT, py::array::c_style | py::array::forcecast> &z,
      RealT tol) {
    copyArrayToVector(z, *zp_);
    copyArrayToVector(v, *v_);
    objective_->update(*zp_, true, -1);
    hv_->zero();
    objective_->hessVec(*hv_, *v_, *zp_, tol);
    return hv_->dot(*v_);
  }

protected:
  virtual void buildModel() = 0;

  void finishBuild() {
    nullStreamHolder_ = std::make_shared<ROL::nullstream>();
    outStream_ = ROL::makePtrFromRef(*nullStreamHolder_);
    meshMgr_ = ROL::makePtr<MeshReader<RealT>>(*parlist_);
    buildModel();

    if (zp_ == ROL::nullPtr || objective_ == ROL::nullPtr) {
      throw std::runtime_error("Flow-opt objective was not fully constructed.");
    }
    v_ = zp_->clone();
    g_ = zp_->dual().clone();
    hv_ = zp_->dual().clone();

    ROL::Ptr<TpetraMV> field = mutableField(*zp_);
    if (field == ROL::nullPtr) {
      throw std::runtime_error("Flow-opt control vector has no field block.");
    }
    localFieldSize_ = field->getLocalLength();
    globalFieldSize_ = field->getGlobalLength();

    ROL::Ptr<std::vector<RealT>> params = mutableParameter(*zp_);
    parameterSize_ = params == ROL::nullPtr ? 0 : params->size();
  }

  ROL::Ptr<ROL::Vector<RealT>>
  makeControlVector(const ROL::Ptr<TpetraMV> &fieldData,
                    const ROL::Ptr<PDE<RealT>> &pde) {
    ROL::Ptr<ROL::TpetraMultiVector<RealT>> field =
        ROL::makePtr<PDE_PrimalOptVector<RealT>>(fieldData, pde, assembler_,
                                                 *parlist_);
    if (!useParamVar_) {
      return field;
    }
    if (z0_ == ROL::nullPtr) {
      z0_ = ROL::makePtr<std::vector<RealT>>(2, static_cast<RealT>(0));
      z0p_ = ROL::makePtr<ROL::StdVector<RealT>>(z0_);
    }
    return ROL::makePtr<PDE_OptVector<RealT>>(field, z0p_, rank_);
  }

  void copyArrayToVector(
      const py::array_t<RealT, py::array::c_style | py::array::forcecast> &arr,
      ROL::Vector<RealT> &vec) const {
    py::buffer_info info = arr.request();
    if (info.ndim != 1) {
      throw std::invalid_argument("Expected a one-dimensional NumPy array.");
    }
    if (static_cast<std::size_t>(info.shape[0]) != localSize()) {
      std::ostringstream os;
      os << "Expected " << localSize() << " local entries, got "
         << info.shape[0] << ".";
      throw std::invalid_argument(os.str());
    }
    const RealT *values = static_cast<const RealT *>(info.ptr);

    ROL::Ptr<TpetraMV> field = mutableField(vec);
    if (field == ROL::nullPtr) {
      throw std::runtime_error("Vector has no field block.");
    }
    Teuchos::ArrayRCP<RealT> data = field->getDataNonConst(0);
    for (std::size_t i = 0; i < localFieldSize_; ++i) {
      data[i] = values[i];
    }

    ROL::Ptr<std::vector<RealT>> params = mutableParameter(vec);
    if (params != ROL::nullPtr) {
      for (std::size_t i = 0; i < params->size(); ++i) {
        (*params)[i] = values[localFieldSize_ + i];
      }
    }
  }

  py::array_t<RealT> copyVectorToArray(const ROL::Vector<RealT> &vec) const {
    std::vector<RealT> values(localSize(), static_cast<RealT>(0));
    ROL::Ptr<const TpetraMV> field = constField(vec);
    if (field == ROL::nullPtr) {
      throw std::runtime_error("Vector has no field block.");
    }
    Teuchos::ArrayRCP<const RealT> data = field->getData(0);
    for (std::size_t i = 0; i < localFieldSize_; ++i) {
      values[i] = data[i];
    }

    ROL::Ptr<const std::vector<RealT>> params = constParameter(vec);
    if (params != ROL::nullPtr) {
      for (std::size_t i = 0; i < params->size(); ++i) {
        values[localFieldSize_ + i] = (*params)[i];
      }
    }
    return makeArray(values);
  }

  ROL::Ptr<TpetraMV> mutableField(ROL::Vector<RealT> &vec) const {
    PDE_OptVector<RealT> *opt = dynamic_cast<PDE_OptVector<RealT> *>(&vec);
    if (opt != nullptr && opt->getField() != ROL::nullPtr) {
      return opt->getField()->getVector();
    }
    ROL::TpetraMultiVector<RealT> *tpetra =
        dynamic_cast<ROL::TpetraMultiVector<RealT> *>(&vec);
    if (tpetra != nullptr) {
      return tpetra->getVector();
    }
    return ROL::nullPtr;
  }

  ROL::Ptr<const TpetraMV> constField(const ROL::Vector<RealT> &vec) const {
    const PDE_OptVector<RealT> *opt =
        dynamic_cast<const PDE_OptVector<RealT> *>(&vec);
    if (opt != nullptr && opt->getField() != ROL::nullPtr) {
      return opt->getField()->getVector();
    }
    const ROL::TpetraMultiVector<RealT> *tpetra =
        dynamic_cast<const ROL::TpetraMultiVector<RealT> *>(&vec);
    if (tpetra != nullptr) {
      return tpetra->getVector();
    }
    return ROL::nullPtr;
  }

  ROL::Ptr<std::vector<RealT>> mutableParameter(ROL::Vector<RealT> &vec) const {
    PDE_OptVector<RealT> *opt = dynamic_cast<PDE_OptVector<RealT> *>(&vec);
    if (opt != nullptr && opt->getParameter() != ROL::nullPtr) {
      return opt->getParameter()->getVector();
    }
    return ROL::nullPtr;
  }

  ROL::Ptr<const std::vector<RealT>>
  constParameter(const ROL::Vector<RealT> &vec) const {
    const PDE_OptVector<RealT> *opt =
        dynamic_cast<const PDE_OptVector<RealT> *>(&vec);
    if (opt != nullptr && opt->getParameter() != ROL::nullPtr) {
      return opt->getParameter()->getVector();
    }
    return ROL::nullPtr;
  }

  std::string xmlPath_;
  std::string xmlDir_;
  int rank_;
  int size_;
  bool useParamVar_;
  std::size_t localFieldSize_;
  std::size_t globalFieldSize_;
  std::size_t parameterSize_;

  ROL::Ptr<const Teuchos::Comm<int>> comm_;
  ROL::Ptr<ROL::ParameterList> parlist_;
  std::shared_ptr<ROL::nullstream> nullStreamHolder_;
  ROL::Ptr<std::ostream> outStream_;
  ROL::Ptr<MeshManager<RealT>> meshMgr_;
  ROL::Ptr<Assembler<RealT>> assembler_;
  ROL::Ptr<PDE<RealT>> pde_;
  ROL::Ptr<ROL::Constraint_SimOpt<RealT>> con_;
  ROL::Ptr<ROL::Objective<RealT>> objective_;

  ROL::Ptr<ROL::Vector<RealT>> up_;
  ROL::Ptr<ROL::Vector<RealT>> pp_;
  ROL::Ptr<ROL::Vector<RealT>> zp_;
  ROL::Ptr<ROL::Vector<RealT>> rp_;
  ROL::Ptr<ROL::Vector<RealT>> v_;
  ROL::Ptr<ROL::Vector<RealT>> g_;
  ROL::Ptr<ROL::Vector<RealT>> hv_;

  ROL::Ptr<TpetraMV> uField_;
  ROL::Ptr<TpetraMV> pField_;
  ROL::Ptr<TpetraMV> zField_;
  ROL::Ptr<TpetraMV> rField_;
  ROL::Ptr<std::vector<RealT>> z0_;
  ROL::Ptr<ROL::StdVector<RealT>> z0p_;
};

template <class ClassT> inline void bindFlowOptMethods(py::class_<ClassT> &cls) {
  cls.def_property_readonly("comm_rank",
                            [](const ClassT &self) { return self.commRank(); })
      .def_property_readonly("comm_size",
                             [](const ClassT &self) { return self.commSize(); })
      .def_property_readonly(
          "local_field_size",
          [](const ClassT &self) { return self.localFieldSize(); })
      .def_property_readonly(
          "global_field_size",
          [](const ClassT &self) { return self.globalFieldSize(); })
      .def_property_readonly(
          "parameter_size",
          [](const ClassT &self) { return self.parameterSize(); })
      .def_property_readonly("local_size",
                             [](const ClassT &self) { return self.localSize(); })
      .def_property_readonly(
          "uses_parameter_control",
          [](const ClassT &self) { return self.usesParameterControl(); })
      .def("value",
           [](ClassT &self,
              const py::array_t<RealT, py::array::c_style |
                                           py::array::forcecast> &z,
              RealT tol) { return self.value(z, tol); },
           py::arg("z"), py::arg("tol") = 1e-8)
      .def("gradient",
           [](ClassT &self,
              const py::array_t<RealT, py::array::c_style |
                                           py::array::forcecast> &z,
              RealT tol) { return self.gradient(z, tol); },
           py::arg("z"), py::arg("tol") = 1e-8)
      .def("hess_vec",
           [](ClassT &self,
              const py::array_t<RealT, py::array::c_style |
                                           py::array::forcecast> &v,
              const py::array_t<RealT, py::array::c_style |
                                           py::array::forcecast> &z,
              RealT tol) { return self.hessVec(v, z, tol); },
           py::arg("v"), py::arg("z"), py::arg("tol") = 1e-8)
      .def("value_and_gradient",
           [](ClassT &self,
              const py::array_t<RealT, py::array::c_style |
                                           py::array::forcecast> &z,
              RealT tol) { return self.valueAndGradient(z, tol); },
           py::arg("z"), py::arg("tol") = 1e-8)
      .def("gradient_dot",
           [](ClassT &self,
              const py::array_t<RealT, py::array::c_style |
                                           py::array::forcecast> &z,
              const py::array_t<RealT, py::array::c_style |
                                           py::array::forcecast> &v,
              RealT tol) { return self.gradientDot(z, v, tol); },
           py::arg("z"), py::arg("v"), py::arg("tol") = 1e-8)
      .def("hess_vec_dot",
           [](ClassT &self,
              const py::array_t<RealT, py::array::c_style |
                                           py::array::forcecast> &v,
              const py::array_t<RealT, py::array::c_style |
                                           py::array::forcecast> &z,
              RealT tol) { return self.hessVecDot(v, z, tol); },
           py::arg("v"), py::arg("z"), py::arg("tol") = 1e-8);
}

void bindDarcy(py::module_ &m);
void bindBrinkman(py::module_ &m);
void bindFilteredDarcy(py::module_ &m);

} // namespace pyrol_flow_opt

#endif // PYROL_FLOW_OPT_COMMON_HPP
