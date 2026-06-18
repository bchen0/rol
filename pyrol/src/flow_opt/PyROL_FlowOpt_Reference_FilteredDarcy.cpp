#include "PyROL_FlowOpt_Reference_Common.hpp"

#include "example/PDE-OPT/flow-opt/axisymmetric/models/filteredDarcy/filtered_obj.hpp"
#include "example/PDE-OPT/flow-opt/axisymmetric/models/filteredDarcy/obj_darcy.hpp"
#include "example/PDE-OPT/flow-opt/axisymmetric/models/filteredDarcy/pde_darcy.hpp"
#include "example/PDE-OPT/flow-opt/axisymmetric/models/filteredDarcy/pde_filter.hpp"

using namespace pyrol_flow_opt_reference;

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " input.xml z.txt v.txt\n";
    return 2;
  }

  try {
    ensureTpetraInitialized();
    ROL::Ptr<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();

    ROL::nullstream bhs;
    ROL::Ptr<std::ostream> outStream = ROL::makePtrFromRef(bhs);
    ROL::Ptr<ROL::ParameterList> parlist = readParameterList(argv[1]);

    ROL::Ptr<MeshManager<RealT>> meshMgr =
        ROL::makePtr<MeshReader<RealT>>(*parlist);
    ROL::Ptr<PDE<RealT>> pde = ROL::makePtr<PDE_Darcy<RealT>>(*parlist);
    ROL::Ptr<ROL::Constraint_SimOpt<RealT>> con =
        ROL::makePtr<PDE_Constraint<RealT>>(pde, meshMgr, comm, *parlist,
                                            *outStream);
    ROL::Ptr<PDE<RealT>> pdeFilter =
        ROL::makePtr<PDE_Filter<RealT>>(*parlist);
    ROL::Ptr<PDE_Constraint<RealT>> pdecon =
        ROL::dynamicPtrCast<PDE_Constraint<RealT>>(con);
    ROL::Ptr<Assembler<RealT>> assembler = pdecon->getAssembler();
    con->setSolveParameters(*parlist);

    ROL::Ptr<TpetraMV> uPtr = assembler->createStateVector();
    ROL::Ptr<TpetraMV> pPtr = assembler->createStateVector();
    ROL::Ptr<TpetraMV> fPtr = assembler->createControlVector();
    ROL::Ptr<TpetraMV> rPtr = assembler->createResidualVector();
    uPtr->randomize();
    pPtr->randomize();
    fPtr->randomize();
    rPtr->putScalar(0.0);

    ROL::Ptr<std::vector<RealT>> z0;
    ROL::Ptr<ROL::StdVector<RealT>> z0p;
    ROL::Ptr<ROL::Vector<RealT>> up =
        ROL::makePtr<PDE_PrimalSimVector<RealT>>(uPtr, pde, assembler,
                                                 *parlist);
    ROL::Ptr<ROL::Vector<RealT>> pp =
        ROL::makePtr<PDE_PrimalSimVector<RealT>>(pPtr, pde, assembler,
                                                 *parlist);
    ROL::Ptr<ROL::Vector<RealT>> fp =
        makeControlVector(fPtr, pde, assembler, *parlist, z0, z0p, rank);

    ROL::Ptr<PDE_Darcy<RealT>> darcy =
        ROL::staticPtrCast<PDE_Darcy<RealT>>(pde);
    ROL::Ptr<QoI<RealT>> qoi =
        ROL::makePtr<QoI_VelocityTracking_Darcy<RealT>>(
            *parlist, darcy->getPressureFE(), darcy->getControlFE(),
            darcy->getPermeability());
    ROL::Ptr<ROL::Objective_SimOpt<RealT>> obj =
        ROL::makePtr<PDE_Objective<RealT>>(qoi, assembler);
    ROL::Ptr<ROL::Reduced_Objective_SimOpt<RealT>> robj =
        ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(
            obj, con, up, fp, pp, true, false);
    ROL::Ptr<FilteredObjective<RealT>> fobj =
        ROL::makePtr<FilteredObjective<RealT>>(robj, pdeFilter, meshMgr, comm,
                                               *parlist, *outStream);

    ROL::Ptr<TpetraMV> zPtr = fobj->getAssembler()->createControlVector();
    zPtr->randomize();
    ROL::Ptr<ROL::TpetraMultiVector<RealT>> densityField =
        ROL::makePtr<PDE_PrimalOptVector<RealT>>(
            zPtr, pdeFilter, fobj->getAssembler(), *parlist);

    ROL::Ptr<ROL::Vector<RealT>> zp;
    const bool useParamVar =
        parlist->sublist("Problem").get("Use Optimal Constant Velocity", false);
    if (useParamVar) {
      zp = ROL::makePtr<PDE_OptVector<RealT>>(densityField, z0p, rank);
    } else {
      zp = densityField;
    }

    return evaluateAndPrint(fobj, zp, argv[2], argv[3]);
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
