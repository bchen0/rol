// Small oracle probe for the ROL dynamic Navier-Stokes reduced objective.
//
// Run from the ROL dynamic/navier-stokes build directory so input.xml and
// channel.txt resolve the same way as the stock example.  The probe overrides
// the time discretization to a tiny run by default, uses the same potential-flow
// initial condition as the Python port, and prints JSON:
//   {"value": ..., "gradient": [...], "hess_vec": [...]}

#include "Teuchos_Comm.hpp"
#include "ROL_GlobalMPISession.hpp"
#include "Tpetra_Core.hpp"

#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "ROL_ParameterList.hpp"
#include "ROL_PartitionedVector.hpp"
#include "ROL_ReducedDynamicObjective.hpp"
#include "ROL_Stream.hpp"

#include "dynpde_navier-stokes.hpp"
#include "initial_condition.hpp"
#include "obj_navier-stokes.hpp"

#include "../../TOOLS/dynconstraint.hpp"
#include "../../TOOLS/ltiobjective.hpp"
#include "../../TOOLS/meshreader.hpp"
#include "../../TOOLS/pdeobjective.hpp"
#include "../../TOOLS/pdevector.hpp"

namespace {

template <class Real>
Real defaultControl(const int k, const int nt) {
  if (k == 0 || nt <= 1) {
    return static_cast<Real>(0);
  }
  if (nt == 2) {
    return static_cast<Real>(0.015);
  }
  const Real a = static_cast<Real>(0.015);
  const Real b = static_cast<Real>(-0.025);
  const Real t = static_cast<Real>(k - 1) / static_cast<Real>(nt - 2);
  return a + t * (b - a);
}

template <class Real>
Real defaultDirection(const int k) {
  return k == 0 ? static_cast<Real>(0) : static_cast<Real>(0.1) * std::cos(static_cast<Real>(k));
}

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
void setScalarControl(ROL::PartitionedVector<Real> &x, const int k, const Real value) {
  ROL::Ptr<ROL::StdVector<Real>> param =
      ROL::dynamicPtrCast<PDE_OptVector<Real>>(x.get(k))->getParameter();
  (*param->getVector())[0] = value;
}

template <class Real>
Real getScalarControl(const ROL::PartitionedVector<Real> &x, const int k) {
  ROL::Ptr<const ROL::StdVector<Real>> param =
      ROL::dynamicPtrCast<const PDE_OptVector<Real>>(x.get(k))->getParameter();
  return (*param->getVector())[0];
}

template <class Real>
std::vector<Real> scalarBlocks(const ROL::Vector<Real> &x, const int nt) {
  const ROL::PartitionedVector<Real> &xp = dynamic_cast<const ROL::PartitionedVector<Real> &>(x);
  std::vector<Real> out(nt, static_cast<Real>(0));
  for (int k = 0; k < nt; ++k) {
    out[k] = getScalarControl(xp, k);
  }
  return out;
}

template <class Real>
void printJson(const Real value,
               const std::vector<Real> &gradient,
               const std::vector<Real> &hessVec) {
  std::cout << std::scientific << std::setprecision(17);
  std::cout << "{\"value\":" << value << ",\"gradient\":[";
  for (std::size_t i = 0; i < gradient.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << gradient[i];
  }
  std::cout << "],\"hess_vec\":[";
  for (std::size_t i = 0; i < hessVec.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << hessVec[i];
  }
  std::cout << "]}" << std::endl;
}

template <class Real>
void printVector(std::ostream &os, const std::vector<Real> &values) {
  os << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << values[i];
  }
  os << "]";
}

template <class Real>
void printTpetraVector(std::ostream &os,
                       const Tpetra::MultiVector<Real> &values,
                       const std::string &name) {
  const auto map = values.getMap();
  const Teuchos::ArrayRCP<const Real> data = values.getData(0);
  const std::size_t n = values.getLocalLength();
  os << "\"" << name << "_gids\":[";
  for (std::size_t i = 0; i < n; ++i) {
    if (i != 0) {
      os << ",";
    }
    os << map->getGlobalElement(static_cast<int>(i));
  }
  os << "],\"" << name << "\":[";
  for (std::size_t i = 0; i < n; ++i) {
    if (i != 0) {
      os << ",";
    }
    os << data[static_cast<int>(i)];
  }
  os << "]";
}

template <class Real>
void printFieldDofs(std::ostream &os, const ROL::Ptr<DofManager<Real>> &dofManager) {
  os << "\"field_dofs\":[";
  for (int field = 0; field < dofManager->getNumFields(); ++field) {
    if (field != 0) {
      os << ",";
    }
    const ROL::Ptr<Intrepid::FieldContainer<int>> dofs = dofManager->getFieldDofs(field);
    const int c = dofs->dimension(0);
    const int f = dofs->dimension(1);
    os << "{\"rows\":" << c << ",\"cols\":" << f << ",\"values\":[";
    for (int i = 0; i < c; ++i) {
      if (i != 0) {
        os << ",";
      }
      os << "[";
      for (int j = 0; j < f; ++j) {
        if (j != 0) {
          os << ",";
        }
        os << (*dofs)(i, j);
      }
      os << "]";
    }
    os << "]}";
  }
  os << "]";
}

template <class Real>
void dumpStateFile(const std::string &path,
                   const ROL::Ptr<Assembler<Real>> &assembler,
                   const Tpetra::MultiVector<Real> &u0,
                   const Tpetra::MultiVector<Real> &un) {
  std::ofstream os(path);
  if (!os) {
    throw std::runtime_error("Could not open --dump-state output file: " + path);
  }
  os << std::scientific << std::setprecision(17);
  os << "{";
  const Teuchos::Array<typename Tpetra::Map<>::global_ordinal_type> cellIds = assembler->getCellIds();
  os << "\"cell_ids\":[";
  for (int i = 0; i < cellIds.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << cellIds[i];
  }
  os << "],";
  printFieldDofs<Real>(os, assembler->getDofManager());
  os << ",";
  printTpetraVector<Real>(os, u0, "u0");
  os << ",";
  printTpetraVector<Real>(os, un, "un");
  os << "}\n";
}

template <class Real>
Real scalarObjectiveValue(const ROL::Ptr<QoI<Real>> &qoi,
                          const ROL::Ptr<Assembler<Real>> &assembler,
                          const ROL::Vector<Real> &u,
                          const ROL::Vector<Real> &z,
                          Real &tol) {
  ROL::Ptr<ROL::Objective_SimOpt<Real>> obj =
      ROL::makePtr<PDE_Objective<Real>>(qoi, assembler);
  obj->update(u, z);
  return obj->value(u, z, tol);
}

} // namespace

int main(int argc, char *argv[]) {
  using RealT = double;

  int ntOverride = 2;
  RealT endTimeOverride = 0.01;
  bool details = false;
  std::string dumpStatePath;
  std::vector<RealT> controlOverride;
  std::vector<RealT> directionOverride;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--time-steps" && i + 1 < argc) {
      ntOverride = std::stoi(argv[++i]);
    } else if (arg == "--end-time" && i + 1 < argc) {
      endTimeOverride = std::stod(argv[++i]);
    } else if (arg == "--details") {
      details = true;
    } else if (arg == "--dump-state" && i + 1 < argc) {
      dumpStatePath = argv[++i];
    } else if (arg == "--control" && i + 1 < argc) {
      controlOverride = parseVector<RealT>(argv[++i]);
    } else if (arg == "--direction" && i + 1 < argc) {
      directionOverride = parseVector<RealT>(argv[++i]);
    }
  }

  ROL::GlobalMPISession mpiSession(&argc, &argv);
  ROL::Ptr<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
  const int numProcs = (comm->getSize() > 1) ? comm->getSize() : 0;
  const int myRank = comm->getRank();
  std::ostringstream sink;
  ROL::Ptr<std::ostream> outStream = ROL::makePtrFromRef<std::ostream>(sink);

  ROL::Ptr<ROL::ParameterList> parlist = ROL::getParametersFromXmlFile("input.xml");
  parlist->sublist("Time Discretization").set("Number of Time Steps", ntOverride);
  parlist->sublist("Time Discretization").set("End Time", endTimeOverride);
  parlist->sublist("Problem").set("Check Derivatives", false);
  parlist->sublist("Problem").set("Print Uncontrolled State", false);
  parlist->sublist("General").set("Print Verbosity", 0);
  parlist->sublist("Dynamic Constraint").sublist("Solve").set("Output Iteration History", false);
  parlist->sublist("SimOpt").sublist("Solve").set("Output Iteration History", false);

  const int nt = parlist->sublist("Time Discretization").get("Number of Time Steps", 100);
  const RealT T = parlist->sublist("Time Discretization").get("End Time", 1.0);
  const RealT dt = T / static_cast<RealT>(nt);
  const bool useParametricControl = parlist->sublist("Problem").get("Use Parametric Control", false);
  if (!useParametricControl) {
    throw std::runtime_error("This probe only supports the default parametric-control branch.");
  }

  parlist->sublist("Reduced Dynamic Objective").set("State Domain Seed", 12321 * (myRank + 1));
  parlist->sublist("Reduced Dynamic Objective").set("State Range Seed", 32123 * (myRank + 1));
  parlist->sublist("Reduced Dynamic Objective").set("Adjoint Domain Seed", 23432 * (myRank + 1));
  parlist->sublist("Reduced Dynamic Objective").set("Adjoint Range Seed", 43234 * (myRank + 1));
  parlist->sublist("Reduced Dynamic Objective").set("State Sensitivity Domain Seed", 34543 * (myRank + 1));
  parlist->sublist("Reduced Dynamic Objective").set("State Sensitivity Range Seed", 54345 * (myRank + 1));

  ROL::Ptr<MeshManager<RealT>> meshMgr = ROL::makePtr<MeshReader<RealT>>(*parlist, numProcs);
  ROL::Ptr<DynamicPDE_NavierStokes<RealT>> pde =
      ROL::makePtr<DynamicPDE_NavierStokes<RealT>>(*parlist);
  ROL::Ptr<DynConstraint<RealT>> dynCon =
      ROL::makePtr<DynConstraint<RealT>>(pde, meshMgr, comm, *parlist, *outStream);
  const ROL::Ptr<Assembler<RealT>> assembler = dynCon->getAssembler();
  dynCon->setSolveParameters(*parlist);

  ROL::Ptr<Tpetra::MultiVector<>> u0Ptr = assembler->createStateVector();
  ROL::Ptr<Tpetra::MultiVector<>> uoPtr = assembler->createStateVector();
  ROL::Ptr<Tpetra::MultiVector<>> unPtr = assembler->createStateVector();
  ROL::Ptr<Tpetra::MultiVector<>> ckPtr = assembler->createResidualVector();

  ROL::Ptr<ROL::Vector<RealT>> u0 =
      ROL::makePtr<PDE_PrimalSimVector<RealT>>(u0Ptr, pde, *assembler, *parlist);
  ROL::Ptr<ROL::Vector<RealT>> uo =
      ROL::makePtr<PDE_PrimalSimVector<RealT>>(uoPtr, pde, *assembler, *parlist);
  ROL::Ptr<ROL::Vector<RealT>> un =
      ROL::makePtr<PDE_PrimalSimVector<RealT>>(unPtr, pde, *assembler, *parlist);
  ROL::Ptr<ROL::Vector<RealT>> ck =
      ROL::makePtr<PDE_DualSimVector<RealT>>(ckPtr, pde, *assembler, *parlist);
  ROL::Ptr<ROL::Vector<RealT>> zk =
      ROL::makePtr<PDE_OptVector<RealT>>(ROL::makePtr<ROL::StdVector<RealT>>(1));
  ROL::Ptr<ROL::PartitionedVector<RealT>> z =
      ROL::PartitionedVector<RealT>::create(*zk, nt);
  ROL::Ptr<ROL::PartitionedVector<RealT>> direction =
      ROL::PartitionedVector<RealT>::create(*zk, nt);

  if (!controlOverride.empty() && static_cast<int>(controlOverride.size()) != nt) {
    throw std::runtime_error("--control length must match --time-steps");
  }
  if (!directionOverride.empty() && static_cast<int>(directionOverride.size()) != nt) {
    throw std::runtime_error("--direction length must match --time-steps");
  }
  for (int k = 0; k < nt; ++k) {
    setScalarControl(*z, k, controlOverride.empty() ? defaultControl<RealT>(k, nt) : controlOverride[k]);
    setScalarControl(*direction, k, directionOverride.empty() ? defaultDirection<RealT>(k) : directionOverride[k]);
  }

  std::vector<ROL::Ptr<QoI<RealT>>> qoiVec(3, ROL::nullPtr), qoiT(1, ROL::nullPtr);
  const RealT w1 = parlist->sublist("Problem").get("State Cost", 1.0);
  const RealT w2 = parlist->sublist("Problem").get("State Boundary Cost", 1.0);
  const RealT w3 = parlist->sublist("Problem").get("Control Cost", 0.0);
  const RealT wT = parlist->sublist("Problem").get("Final Time State Cost", 1.0);
  std::vector<RealT> wts = {w1, w2, w3}, wtsT = {wT};
  const std::string intObj =
      parlist->sublist("Problem").get("Integrated Objective Type", "Dissipation");
  const std::string ftObj =
      parlist->sublist("Problem").get("Final Time Objective Type", "Tracking");
  qoiVec[0] = ROL::makePtr<QoI_State_NavierStokes<RealT>>(
      intObj, *parlist, pde->getVelocityFE(), pde->getPressureFE(), pde->getFieldHelper());
  qoiVec[1] = ROL::makePtr<QoI_DownStreamPower_NavierStokes<RealT>>(
      pde->getVelocityFE(), pde->getPressureFE(), pde->getVelocityBdryFE(1),
      pde->getBdryCellLocIds(1), pde->getFieldHelper());
  qoiVec[2] = ROL::makePtr<QoI_RotationControl_NavierStokes<RealT>>();
  qoiT[0] = ROL::makePtr<QoI_State_NavierStokes<RealT>>(
      ftObj, *parlist, pde->getVelocityFE(), pde->getPressureFE(), pde->getFieldHelper());

  ROL::Ptr<ROL::Objective_SimOpt<RealT>> objK =
      ROL::makePtr<PDE_Objective<RealT>>(qoiVec, wts, assembler);
  ROL::Ptr<ROL::Objective_SimOpt<RealT>> objT =
      ROL::makePtr<PDE_Objective<RealT>>(qoiT, wtsT, assembler);
  ROL::Ptr<LTI_Objective<RealT>> dynObj =
      ROL::makePtr<LTI_Objective<RealT>>(*parlist, objK, objT);

  std::vector<ROL::TimeStamp<RealT>> timeStamp(nt);
  for (int k = 0; k < nt; ++k) {
    timeStamp.at(k).t.resize(2);
    timeStamp.at(k).t.at(0) = k * dt;
    timeStamp.at(k).t.at(1) = (k + 1) * dt;
  }

  PotentialFlow<RealT> potentialFlow(pde->getVelocityFE(), pde->getPressureFE(), pde->getCellNodes(),
                                     assembler->getDofManager()->getCellDofs(),
                                     assembler->getCellIds(), pde->getFieldHelper(), *parlist);
  potentialFlow.build(u0Ptr);

  ROL::ParameterList &rpl = parlist->sublist("Reduced Dynamic Objective");
  ROL::Ptr<ROL::ReducedDynamicObjective<RealT>> obj =
      ROL::makePtr<ROL::ReducedDynamicObjective<RealT>>(dynObj, dynCon, u0, zk, ck, timeStamp, rpl, outStream);

  RealT tol = 1.e-12;
  const RealT value = obj->value(*z, tol);
  ROL::Ptr<ROL::Vector<RealT>> gradient = z->dual().clone();
  gradient->zero();
  obj->gradient(*gradient, *z, tol);
  ROL::Ptr<ROL::Vector<RealT>> hessVec = direction->dual().clone();
  hessVec->zero();
  obj->hessVec(*hessVec, *direction, *z, tol);

  if (myRank == 0) {
    const std::vector<RealT> gradBlocks = scalarBlocks(*gradient, nt);
    const std::vector<RealT> hessBlocks = scalarBlocks(*hessVec, nt);
    if (!details) {
      printJson(value, gradBlocks, hessBlocks);
    } else {
      uo->set(*u0);
      un->set(*u0);
      dynCon->update_uo(*uo, timeStamp[1]);
      dynCon->update_z(*z->get(1), timeStamp[1]);
      dynCon->solve(*ck, *uo, *un, *z->get(1), timeStamp[1]);
      const RealT residualNorm = ck->norm();
      if (!dumpStatePath.empty()) {
        dumpStateFile<RealT>(dumpStatePath, assembler, *u0Ptr, *unPtr);
      }

      std::vector<RealT> oldComponents, newComponents;
      for (std::size_t i = 0; i < qoiVec.size(); ++i) {
        oldComponents.push_back(scalarObjectiveValue<RealT>(qoiVec[i], assembler, *uo, *z->get(1), tol));
        newComponents.push_back(scalarObjectiveValue<RealT>(qoiVec[i], assembler, *un, *z->get(1), tol));
      }
      const RealT finalComponent = scalarObjectiveValue<RealT>(qoiT[0], assembler, *un, *z->get(1), tol);
      const RealT weightedOld = w1 * oldComponents[0] + w2 * oldComponents[1] + w3 * oldComponents[2];
      const RealT weightedNew = w1 * newComponents[0] + w2 * newComponents[1] + w3 * newComponents[2];
      const RealT manualValue = dt * ((static_cast<RealT>(1) - parlist->sublist("Time Discretization").get("Theta", 1.0)) * weightedOld
                            + parlist->sublist("Time Discretization").get("Theta", 1.0) * weightedNew)
                            + wT * finalComponent;
      std::cout << std::scientific << std::setprecision(17);
      std::cout << "{\"value\":" << value << ",\"gradient\":";
      printVector(std::cout, gradBlocks);
      std::cout << ",\"hess_vec\":";
      printVector(std::cout, hessBlocks);
      std::cout << ",\"details\":{\"qoi_names\":[\"state\",\"downstream\",\"control\"],\"old_components\":";
      printVector(std::cout, oldComponents);
      std::cout << ",\"new_components\":";
      printVector(std::cout, newComponents);
      std::cout << ",\"final_component\":" << finalComponent
                << ",\"weighted_old\":" << weightedOld
                << ",\"weighted_new\":" << weightedNew
                << ",\"manual_value\":" << manualValue
                << ",\"u0_norm\":" << u0->norm()
                << ",\"un_norm\":" << un->norm()
                << ",\"residual_norm\":" << residualNorm
                << "}}" << std::endl;
    }
  }
  return 0;
}
