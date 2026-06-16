// Oracle probe for the ROL axisymmetric filtered Darcy objective.
//
// The probe constructs the same object as filteredDarcy/example_01.cpp:
//   fobj = FilteredObjective(Reduced_Objective_SimOpt(PDE_Objective(QoI), PDE_Darcy), PDE_Filter)
// and prints JSON with value, gradient, and Hessian-vector product. With
// --dump-check-vectors it instead emits the randomized rzp/dzp vectors used by
// example_01.cpp for fobj->checkGradient and fobj->checkHessVec.

#include "Teuchos_Comm.hpp"
#include "ROL_GlobalMPISession.hpp"
#include "Tpetra_Core.hpp"

#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ROL_ParameterList.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_Stream.hpp"
#include "ROL_TpetraMultiVector.hpp"

#include "../../../../TOOLS/meshreader.hpp"
#include "../../../../TOOLS/pdeconstraint.hpp"
#include "../../../../TOOLS/pdeobjective.hpp"
#include "../../../../TOOLS/pdevector.hpp"

#include "filtered_obj.hpp"
#include "obj_darcy.hpp"
#include "pde_darcy.hpp"
#include "pde_filter.hpp"

namespace {

template <class Real>
std::vector<Real> parseVector(const std::string &text) {
  std::string cleaned;
  cleaned.reserve(text.size());
  for (char ch : text) {
    if (ch == '[' || ch == ']' || std::isspace(static_cast<unsigned char>(ch))) {
      continue;
    }
    cleaned.push_back(ch == ';' ? ',' : ch);
  }
  std::vector<Real> out;
  std::stringstream stream(cleaned);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      out.push_back(static_cast<Real>(std::stod(token)));
    }
  }
  return out;
}

template <class Real>
void fillTpetraVector(const ROL::Ptr<Tpetra::MultiVector<>> &vec,
                      const std::vector<Real> &values,
                      const std::size_t offset) {
  Teuchos::ArrayRCP<Real> data = vec->getDataNonConst(0);
  const std::size_t n = data.size();
  if (offset + n > values.size()) {
    throw std::runtime_error("Input vector is too short for Tpetra field block");
  }
  for (std::size_t i = 0; i < n; ++i) {
    data[static_cast<int>(i)] = values[offset + i];
  }
}

template <class Real>
void fillStdVector(const ROL::Ptr<std::vector<Real>> &vec,
                   const std::vector<Real> &values,
                   const std::size_t offset) {
  if (vec == ROL::nullPtr) {
    return;
  }
  if (offset + vec->size() > values.size()) {
    throw std::runtime_error("Input vector is too short for parameter block");
  }
  for (std::size_t i = 0; i < vec->size(); ++i) {
    (*vec)[i] = values[offset + i];
  }
}

template <class Real>
ROL::Ptr<const Tpetra::MultiVector<>> getConstField(const ROL::Vector<Real> &x) {
  try {
    return dynamic_cast<const ROL::TpetraMultiVector<Real>&>(x).getVector();
  }
  catch (std::exception &) {
    ROL::Ptr<const ROL::TpetraMultiVector<Real>> field =
        dynamic_cast<const PDE_OptVector<Real>&>(x).getField();
    return field == ROL::nullPtr ? ROL::nullPtr : field->getVector();
  }
}

template <class Real>
ROL::Ptr<Tpetra::MultiVector<>> getField(ROL::Vector<Real> &x) {
  try {
    return dynamic_cast<ROL::TpetraMultiVector<Real>&>(x).getVector();
  }
  catch (std::exception &) {
    ROL::Ptr<ROL::TpetraMultiVector<Real>> field =
        dynamic_cast<PDE_OptVector<Real>&>(x).getField();
    return field == ROL::nullPtr ? ROL::nullPtr : field->getVector();
  }
}

template <class Real>
ROL::Ptr<const std::vector<Real>> getConstParameter(const ROL::Vector<Real> &x) {
  try {
    return dynamic_cast<const ROL::StdVector<Real>&>(x).getVector();
  }
  catch (std::exception &) {
    ROL::Ptr<const ROL::StdVector<Real>> param =
        dynamic_cast<const PDE_OptVector<Real>&>(x).getParameter();
    return param == ROL::nullPtr ? ROL::nullPtr : param->getVector();
  }
}

template <class Real>
std::vector<Real> flattenVector(const ROL::Vector<Real> &x) {
  std::vector<Real> out;
  ROL::Ptr<const Tpetra::MultiVector<>> field = getConstField(x);
  if (field != ROL::nullPtr) {
    Teuchos::ArrayRCP<const Real> data = field->getData(0);
    for (int i = 0; i < data.size(); ++i) {
      out.push_back(data[i]);
    }
  }
  ROL::Ptr<const std::vector<Real>> param = getConstParameter(x);
  if (param != ROL::nullPtr) {
    out.insert(out.end(), param->begin(), param->end());
  }
  return out;
}

template <class Real>
std::vector<Real> flattenField(const ROL::Vector<Real> &x) {
  std::vector<Real> out;
  ROL::Ptr<const Tpetra::MultiVector<>> field = getConstField(x);
  if (field != ROL::nullPtr) {
    Teuchos::ArrayRCP<const Real> data = field->getData(0);
    for (int i = 0; i < data.size(); ++i) {
      out.push_back(data[i]);
    }
  }
  return out;
}

template <class Real>
std::vector<Real> flattenParameter(const ROL::Vector<Real> &x) {
  std::vector<Real> out;
  ROL::Ptr<const std::vector<Real>> param = getConstParameter(x);
  if (param != ROL::nullPtr) {
    out.insert(out.end(), param->begin(), param->end());
  }
  return out;
}

template <class Real>
void printVector(const std::vector<Real> &values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

} // namespace

int main(int argc, char *argv[]) {
  using RealT = double;

  std::string inputPath = "input.xml";
  std::string meshPath;
  std::vector<RealT> xOverride;
  std::vector<RealT> vOverride;
  bool dumpCheckVectors = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--input" && i + 1 < argc) {
      inputPath = argv[++i];
    }
    else if (arg == "--mesh" && i + 1 < argc) {
      meshPath = argv[++i];
    }
    else if (arg == "--x" && i + 1 < argc) {
      xOverride = parseVector<RealT>(argv[++i]);
    }
    else if (arg == "--v" && i + 1 < argc) {
      vOverride = parseVector<RealT>(argv[++i]);
    }
    else if (arg == "--dump-check-vectors") {
      dumpCheckVectors = true;
    }
  }

  ROL::GlobalMPISession mpiSession(&argc, &argv);
  ROL::nullstream bhs;
  ROL::Ptr<std::ostream> outStream = ROL::makePtrFromRef(bhs);
  ROL::Ptr<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
  const int myRank = comm->getRank();

  try {
    ROL::Ptr<Teuchos::ParameterList> parlist = ROL::makePtr<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile(inputPath, parlist.ptr());
    if (!meshPath.empty()) {
      parlist->sublist("Mesh").set("File Name", meshPath);
    }
    parlist->sublist("Problem").set("Solve Optimization Problem", false);
    parlist->sublist("Problem").set("Check derivatives", false);

    ROL::Ptr<MeshManager<RealT>> meshMgr = ROL::makePtr<MeshReader<RealT>>(*parlist);
    ROL::Ptr<PDE<RealT>> pde = ROL::makePtr<PDE_Darcy<RealT>>(*parlist);
    ROL::Ptr<ROL::Constraint_SimOpt<RealT>> con =
        ROL::makePtr<PDE_Constraint<RealT>>(pde, meshMgr, comm, *parlist, *outStream);
    ROL::Ptr<PDE<RealT>> pdeFilter = ROL::makePtr<PDE_Filter<RealT>>(*parlist);
    ROL::Ptr<PDE_Constraint<RealT>> pdecon = ROL::dynamicPtrCast<PDE_Constraint<RealT>>(con);
    ROL::Ptr<Assembler<RealT>> assembler = pdecon->getAssembler();
    con->setSolveParameters(*parlist);

    ROL::Ptr<Tpetra::MultiVector<>> u_ptr = assembler->createStateVector();
    ROL::Ptr<Tpetra::MultiVector<>> p_ptr = assembler->createStateVector();
    ROL::Ptr<Tpetra::MultiVector<>> f_ptr = assembler->createControlVector();
    if (dumpCheckVectors) {
      u_ptr->randomize();
      p_ptr->randomize();
      f_ptr->randomize();
    }
    else {
      u_ptr->putScalar(0.0);
      p_ptr->putScalar(0.0);
      f_ptr->putScalar(0.0);
    }

    const bool useParamVar = parlist->sublist("Problem").get("Use Optimal Constant Velocity", false);
    const int dim = 2;
    ROL::Ptr<std::vector<RealT>> z0_ptr;
    ROL::Ptr<ROL::StdVector<RealT>> z0p;
    ROL::Ptr<ROL::Vector<RealT>> fp;
    if (useParamVar) {
      z0_ptr = ROL::makePtr<std::vector<RealT>>(dim, static_cast<RealT>(0));
      z0p = ROL::makePtr<ROL::StdVector<RealT>>(z0_ptr);
      ROL::Ptr<ROL::TpetraMultiVector<RealT>> f1p =
          ROL::makePtr<PDE_PrimalOptVector<RealT>>(f_ptr, pde, assembler, *parlist);
      fp = ROL::makePtr<PDE_OptVector<RealT>>(f1p, z0p, myRank);
    }
    else {
      fp = ROL::makePtr<PDE_PrimalOptVector<RealT>>(f_ptr, pde, assembler, *parlist);
    }
    ROL::Ptr<ROL::Vector<RealT>> up =
        ROL::makePtr<PDE_PrimalSimVector<RealT>>(u_ptr, pde, assembler, *parlist);
    ROL::Ptr<ROL::Vector<RealT>> pp =
        ROL::makePtr<PDE_PrimalSimVector<RealT>>(p_ptr, pde, assembler, *parlist);

    ROL::Ptr<QoI<RealT>> qoi = ROL::makePtr<QoI_VelocityTracking_Darcy<RealT>>(
        *parlist,
        ROL::staticPtrCast<PDE_Darcy<RealT>>(pde)->getPressureFE(),
        ROL::staticPtrCast<PDE_Darcy<RealT>>(pde)->getControlFE(),
        ROL::staticPtrCast<PDE_Darcy<RealT>>(pde)->getPermeability());
    ROL::Ptr<ROL::Objective_SimOpt<RealT>> obj =
        ROL::makePtr<PDE_Objective<RealT>>(qoi, assembler);
    ROL::Ptr<ROL::Reduced_Objective_SimOpt<RealT>> robj =
        ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(obj, con, up, fp, pp, true, false);
    ROL::Ptr<FilteredObjective<RealT>> fobj =
        ROL::makePtr<FilteredObjective<RealT>>(robj, pdeFilter, meshMgr, comm, *parlist, *outStream);

    ROL::Ptr<Tpetra::MultiVector<>> z_ptr = fobj->getAssembler()->createControlVector();
    if (dumpCheckVectors) {
      z_ptr->randomize();
    }
    else {
      z_ptr->putScalar(0.0);
    }
    ROL::Ptr<ROL::Vector<RealT>> zp;
    if (useParamVar) {
      ROL::Ptr<ROL::TpetraMultiVector<RealT>> z1p =
          ROL::makePtr<PDE_PrimalOptVector<RealT>>(z_ptr, pdeFilter, fobj->getAssembler(), *parlist);
      zp = ROL::makePtr<PDE_OptVector<RealT>>(z1p, z0p, myRank);
    }
    else {
      zp = ROL::makePtr<PDE_PrimalOptVector<RealT>>(z_ptr, pdeFilter, fobj->getAssembler(), *parlist);
    }
    ROL::Ptr<ROL::Vector<RealT>> vp = zp->clone();
    ROL::Ptr<ROL::Vector<RealT>> gp = zp->clone();
    ROL::Ptr<ROL::Vector<RealT>> hvp = zp->clone();

    const std::size_t densitySize = z_ptr->getLocalLength();
    const std::size_t paramSize = useParamVar ? static_cast<std::size_t>(dim) : 0;
    const std::size_t totalSize = densitySize + paramSize;
    if (dumpCheckVectors) {
      ROL::Ptr<ROL::Vector<RealT>> rzp = zp->clone();
      ROL::Ptr<ROL::Vector<RealT>> dzp = zp->clone();
      rzp->randomize(static_cast<RealT>(0), static_cast<RealT>(1));
      dzp->randomize(static_cast<RealT>(0), static_cast<RealT>(1));

      if (myRank == 0) {
        std::cout << std::scientific << std::setprecision(17);
        std::cout << "{\"mode\":\"check_vectors\""
                  << ",\"density_size\":" << densitySize
                  << ",\"parameter_size\":" << paramSize
                  << ",\"rzp\":";
        printVector(flattenVector<RealT>(*rzp));
        std::cout << ",\"dzp\":";
        printVector(flattenVector<RealT>(*dzp));
        std::cout << ",\"rzp_field\":";
        printVector(flattenField<RealT>(*rzp));
        std::cout << ",\"rzp_parameter\":";
        printVector(flattenParameter<RealT>(*rzp));
        std::cout << ",\"dzp_field\":";
        printVector(flattenField<RealT>(*dzp));
        std::cout << ",\"dzp_parameter\":";
        printVector(flattenParameter<RealT>(*dzp));
        std::cout << "}" << std::endl;
      }
      return 0;
    }

    std::vector<RealT> x = xOverride.empty() ? std::vector<RealT>(totalSize, static_cast<RealT>(0.5)) : xOverride;
    std::vector<RealT> v(totalSize, static_cast<RealT>(0));
    if (vOverride.empty()) {
      for (std::size_t i = 0; i < totalSize; ++i) {
        v[i] = static_cast<RealT>(0.1) * static_cast<RealT>(i + 1);
      }
    }
    else {
      v = vOverride;
    }
    if (x.size() != totalSize || v.size() != totalSize) {
      throw std::runtime_error("Input x/v length does not match C++ vector dimension");
    }

    fillTpetraVector<RealT>(z_ptr, x, 0);
    fillStdVector<RealT>(z0_ptr, x, densitySize);
    ROL::Ptr<Tpetra::MultiVector<>> v_ptr = getField<RealT>(*vp);
    fillTpetraVector<RealT>(v_ptr, v, 0);
    if (useParamVar) {
      ROL::Ptr<std::vector<RealT>> vp_param =
          ROL::dynamicPtrCast<PDE_OptVector<RealT>>(vp)->getParameter()->getVector();
      fillStdVector<RealT>(vp_param, v, densitySize);
    }

    RealT tol = static_cast<RealT>(1e-8);
    fobj->update(*zp, ROL::UpdateType::Initial);
    const RealT value = fobj->value(*zp, tol);
    fobj->gradient(*gp, *zp, tol);
    fobj->hessVec(*hvp, *vp, *zp, tol);

    if (myRank == 0) {
      std::cout << std::scientific << std::setprecision(17);
      std::cout << "{\"density_size\":" << densitySize
                << ",\"parameter_size\":" << paramSize
                << ",\"value\":" << value
                << ",\"gradient\":";
      printVector(flattenVector<RealT>(*gp));
      std::cout << ",\"hess_vec\":";
      printVector(flattenVector<RealT>(*hvp));
      std::cout << "}" << std::endl;
    }
  }
  catch (const std::exception &e) {
    if (myRank == 0) {
      std::cerr << "rol_filtered_darcy_probe failed: " << e.what() << std::endl;
    }
    return 1;
  }
  return 0;
}
