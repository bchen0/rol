#ifndef PYROL_FLOW_OPT_REFERENCE_COMMON_HPP
#define PYROL_FLOW_OPT_REFERENCE_COMMON_HPP

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
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyrol_flow_opt_reference {

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

inline std::string rankPath(const std::string &path, const int rank) {
  std::string out = path;
  const std::string token = "{rank}";
  const std::string rankString = std::to_string(rank);
  std::string::size_type pos = 0;
  while ((pos = out.find(token, pos)) != std::string::npos) {
    out.replace(pos, token.size(), rankString);
    pos += rankString.size();
  }
  return out;
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
  if (parlist.isSublist("Geometry")) {
    ROL::ParameterList &geometry = parlist.sublist("Geometry");
    if (geometry.isParameter("Mesh File")) {
      const std::string meshFile = geometry.get<std::string>("Mesh File");
      if (!isAbsolutePath(meshFile)) {
        geometry.set("Mesh File", joinPath(xmlDir, meshFile));
      }
    }
  }
}

inline ROL::Ptr<ROL::ParameterList>
readParameterList(const std::string &xmlPath) {
  ROL::Ptr<ROL::ParameterList> parlist = ROL::makePtr<ROL::ParameterList>();
  Teuchos::updateParametersFromXmlFile(xmlPath, parlist.ptr());
  absolutizeMeshPath(*parlist, dirname(xmlPath));

  if (parlist->isSublist("Problem")) {
    ROL::ParameterList &problem = parlist->sublist("Problem");
    problem.set("Check derivatives", false);
    problem.set("Solve Optimization Problem", false);
  }
  if (parlist->isSublist("SimOpt")) {
    parlist->sublist("SimOpt")
        .sublist("Solve")
        .set("Output Iteration History", false);
  }
  return parlist;
}

inline std::vector<RealT> readVectorFile(const std::string &path) {
  std::ifstream in(path);
  if (!in.good()) {
    throw std::runtime_error("could not open vector file '" + path + "'");
  }
  std::vector<RealT> values;
  RealT value = 0;
  while (in >> value) {
    values.push_back(value);
  }
  return values;
}

inline ROL::Ptr<TpetraMV> mutableField(ROL::Vector<RealT> &vec) {
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

inline ROL::Ptr<const TpetraMV> constField(const ROL::Vector<RealT> &vec) {
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

inline ROL::Ptr<std::vector<RealT>> mutableParameter(ROL::Vector<RealT> &vec) {
  PDE_OptVector<RealT> *opt = dynamic_cast<PDE_OptVector<RealT> *>(&vec);
  if (opt != nullptr && opt->getParameter() != ROL::nullPtr) {
    return opt->getParameter()->getVector();
  }
  return ROL::nullPtr;
}

inline ROL::Ptr<const std::vector<RealT>>
constParameter(const ROL::Vector<RealT> &vec) {
  const PDE_OptVector<RealT> *opt =
      dynamic_cast<const PDE_OptVector<RealT> *>(&vec);
  if (opt != nullptr && opt->getParameter() != ROL::nullPtr) {
    return opt->getParameter()->getVector();
  }
  return ROL::nullPtr;
}

inline std::size_t localFieldSize(const ROL::Vector<RealT> &vec) {
  ROL::Ptr<const TpetraMV> field = constField(vec);
  if (field == ROL::nullPtr) {
    throw std::runtime_error("reference vector has no field block");
  }
  return field->getLocalLength();
}

inline std::size_t parameterSize(const ROL::Vector<RealT> &vec) {
  ROL::Ptr<const std::vector<RealT>> params = constParameter(vec);
  return params == ROL::nullPtr ? 0 : params->size();
}

inline std::size_t localSize(const ROL::Vector<RealT> &vec) {
  return localFieldSize(vec) + parameterSize(vec);
}

inline void copyToVector(const std::vector<RealT> &values,
                         ROL::Vector<RealT> &vec) {
  const std::size_t fieldSize = localFieldSize(vec);
  const std::size_t paramSize = parameterSize(vec);
  if (values.size() != fieldSize + paramSize) {
    std::ostringstream os;
    os << "expected " << fieldSize + paramSize << " vector entries, got "
       << values.size();
    throw std::runtime_error(os.str());
  }

  ROL::Ptr<TpetraMV> field = mutableField(vec);
  Teuchos::ArrayRCP<RealT> data = field->getDataNonConst(0);
  for (std::size_t i = 0; i < fieldSize; ++i) {
    data[i] = values[i];
  }

  ROL::Ptr<std::vector<RealT>> params = mutableParameter(vec);
  if (params != ROL::nullPtr) {
    for (std::size_t i = 0; i < params->size(); ++i) {
      (*params)[i] = values[fieldSize + i];
    }
  }
}

inline std::vector<RealT> copyFromVector(const ROL::Vector<RealT> &vec,
                                         const ROL::Vector<RealT> &shape) {
  const std::size_t fieldSize = localFieldSize(shape);
  const std::size_t paramSize = parameterSize(shape);
  std::vector<RealT> values(fieldSize + paramSize, static_cast<RealT>(0));

  ROL::Ptr<const TpetraMV> field = constField(vec);
  Teuchos::ArrayRCP<const RealT> data = field->getData(0);
  for (std::size_t i = 0; i < fieldSize; ++i) {
    values[i] = data[i];
  }

  ROL::Ptr<const std::vector<RealT>> params = constParameter(vec);
  if (params != ROL::nullPtr) {
    for (std::size_t i = 0; i < params->size(); ++i) {
      values[fieldSize + i] = (*params)[i];
    }
  }
  return values;
}

inline void printVector(const std::string &label,
                        const std::vector<RealT> &values) {
  std::cout << label << " " << values.size();
  std::cout << std::setprecision(17);
  for (RealT value : values) {
    std::cout << " " << value;
  }
  std::cout << "\n";
}

inline int evaluateAndPrint(
    const ROL::Ptr<ROL::Objective<RealT>> &objective,
    const ROL::Ptr<ROL::Vector<RealT>> &z,
    const std::string &zPath,
    const std::string &vPath,
    RealT tol = 1e-8) {
  ROL::Ptr<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
  const int rank = comm->getRank();
  const std::vector<RealT> zValues = readVectorFile(rankPath(zPath, rank));
  const std::vector<RealT> vValues = readVectorFile(rankPath(vPath, rank));

  ROL::Ptr<ROL::Vector<RealT>> v = z->clone();
  ROL::Ptr<ROL::Vector<RealT>> g = z->dual().clone();
  ROL::Ptr<ROL::Vector<RealT>> hv = z->dual().clone();

  copyToVector(zValues, *z);
  copyToVector(vValues, *v);

  auto timed = [&comm](const std::function<void()> &op) {
    comm->barrier();
    const auto start = std::chrono::steady_clock::now();
    op();
    comm->barrier();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<RealT>(end - start).count();
  };

  RealT value = 0;
  const RealT valueSeconds = timed([&]() {
    objective->update(*z, true, -1);
    value = objective->value(*z, tol);
  });
  const RealT gradientSeconds = timed([&]() {
    objective->update(*z, true, -1);
    g->zero();
    objective->gradient(*g, *z, tol);
  });
  const RealT hessVecSeconds = timed([&]() {
    objective->update(*z, true, -1);
    hv->zero();
    objective->hessVec(*hv, *v, *z, tol);
  });

  const std::vector<RealT> gradient = copyFromVector(*g, *z);
  const std::vector<RealT> hessVec = copyFromVector(*hv, *z);

  std::cout << std::setprecision(17);
  std::cout << "LOCAL_SIZE " << localSize(*z) << "\n";
  std::cout << "VALUE " << value << "\n";
  printVector("GRADIENT", gradient);
  printVector("HESS_VEC", hessVec);
  const RealT gradientDot = g->dot(*v);
  const RealT hessVecDot = hv->dot(*v);
  std::cout << "GRADIENT_DOT " << gradientDot << "\n";
  std::cout << "HESS_VEC_DOT " << hessVecDot << "\n";
  std::cout << "TIMES " << valueSeconds << " " << gradientSeconds << " "
            << hessVecSeconds << "\n";
  std::cout << "RESULT " << comm->getRank() << " " << localSize(*z) << " "
            << value << " " << gradientDot << " " << hessVecDot << " "
            << valueSeconds << " " << gradientSeconds << " " << hessVecSeconds
            << " " << gradient.size();
  for (RealT x : gradient) {
    std::cout << " " << x;
  }
  std::cout << " " << hessVec.size();
  for (RealT x : hessVec) {
    std::cout << " " << x;
  }
  std::cout << "\n";
  return 0;
}

inline ROL::Ptr<ROL::Vector<RealT>>
makeControlVector(const ROL::Ptr<TpetraMV> &fieldData,
                  const ROL::Ptr<PDE<RealT>> &pde,
                  const ROL::Ptr<Assembler<RealT>> &assembler,
                  ROL::ParameterList &parlist,
                  ROL::Ptr<std::vector<RealT>> &z0,
                  ROL::Ptr<ROL::StdVector<RealT>> &z0p,
                  const int rank) {
  ROL::Ptr<ROL::TpetraMultiVector<RealT>> field =
      ROL::makePtr<PDE_PrimalOptVector<RealT>>(fieldData, pde, assembler,
                                               parlist);
  const bool useParamVar =
      parlist.sublist("Problem").get("Use Optimal Constant Velocity", false);
  if (!useParamVar) {
    return field;
  }
  if (z0 == ROL::nullPtr) {
    z0 = ROL::makePtr<std::vector<RealT>>(2, static_cast<RealT>(0));
    z0p = ROL::makePtr<ROL::StdVector<RealT>>(z0);
  }
  return ROL::makePtr<PDE_OptVector<RealT>>(field, z0p, rank);
}

} // namespace pyrol_flow_opt_reference

#endif // PYROL_FLOW_OPT_REFERENCE_COMMON_HPP
