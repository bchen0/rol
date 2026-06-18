#include "PyROL_FlowOpt_Common.hpp"

#include "example/PDE-OPT/flow-opt/axisymmetric/models/filteredDarcy/filtered_obj.hpp"
#include "example/PDE-OPT/flow-opt/axisymmetric/models/filteredDarcy/obj_darcy.hpp"
#include "example/PDE-OPT/flow-opt/axisymmetric/models/filteredDarcy/pde_darcy.hpp"
#include "example/PDE-OPT/flow-opt/axisymmetric/models/filteredDarcy/pde_filter.hpp"

namespace pyrol_flow_opt {

namespace {

class FilteredDarcyObjective final : public FlowOptObjectiveBase {
public:
  explicit FilteredDarcyObjective(const std::string &xmlPath)
      : FlowOptObjectiveBase(xmlPath) {
    finishBuild();
  }

protected:
  void buildModel() override {
    pde_ = ROL::makePtr<PDE_Darcy<RealT>>(*parlist_);
    con_ = ROL::makePtr<PDE_Constraint<RealT>>(pde_, meshMgr_, comm_, *parlist_,
                                               *outStream_);
    pdeFilter_ = ROL::makePtr<PDE_Filter<RealT>>(*parlist_);

    ROL::Ptr<PDE_Constraint<RealT>> pdecon =
        ROL::dynamicPtrCast<PDE_Constraint<RealT>>(con_);
    assembler_ = pdecon->getAssembler();
    con_->setSolveParameters(*parlist_);

    uField_ = assembler_->createStateVector();
    pField_ = assembler_->createStateVector();
    filteredField_ = assembler_->createControlVector();
    rField_ = assembler_->createResidualVector();
    uField_->randomize();
    pField_->randomize();
    filteredField_->randomize();
    rField_->putScalar(0.0);

    up_ = ROL::makePtr<PDE_PrimalSimVector<RealT>>(uField_, pde_, assembler_,
                                                   *parlist_);
    pp_ = ROL::makePtr<PDE_PrimalSimVector<RealT>>(pField_, pde_, assembler_,
                                                   *parlist_);
    filteredControl_ = makeControlVector(filteredField_, pde_);
    rp_ = ROL::makePtr<PDE_DualSimVector<RealT>>(rField_, pde_, assembler_,
                                                 *parlist_);

    ROL::Ptr<PDE_Darcy<RealT>> darcy =
        ROL::staticPtrCast<PDE_Darcy<RealT>>(pde_);
    ROL::Ptr<QoI<RealT>> qoi =
        ROL::makePtr<QoI_VelocityTracking_Darcy<RealT>>(
            *parlist_, darcy->getPressureFE(), darcy->getControlFE(),
            darcy->getPermeability());
    ROL::Ptr<ROL::Objective_SimOpt<RealT>> obj =
        ROL::makePtr<PDE_Objective<RealT>>(qoi, assembler_);
    ROL::Ptr<ROL::Reduced_Objective_SimOpt<RealT>> robj =
        ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(
            obj, con_, up_, filteredControl_, pp_, true, false);

    filteredObjective_ = ROL::makePtr<FilteredObjective<RealT>>(
        robj, pdeFilter_, meshMgr_, comm_, *parlist_, *outStream_);
    objective_ = filteredObjective_;

    zField_ = filteredObjective_->getAssembler()->createControlVector();
    zField_->randomize();
    ROL::Ptr<ROL::TpetraMultiVector<RealT>> densityField =
        ROL::makePtr<PDE_PrimalOptVector<RealT>>(
            zField_, pdeFilter_, filteredObjective_->getAssembler(), *parlist_);
    if (useParamVar_) {
      zp_ = ROL::makePtr<PDE_OptVector<RealT>>(densityField, z0p_, rank_);
    } else {
      zp_ = densityField;
    }
  }

private:
  ROL::Ptr<PDE<RealT>> pdeFilter_;
  ROL::Ptr<TpetraMV> filteredField_;
  ROL::Ptr<ROL::Vector<RealT>> filteredControl_;
  ROL::Ptr<FilteredObjective<RealT>> filteredObjective_;
};

} // namespace

void bindFilteredDarcy(py::module_ &m) {
  py::class_<FilteredDarcyObjective> cls(m, "_FilteredDarcyObjective");
  cls.def(py::init<const std::string &>(), py::arg("xml_path"));
  bindFlowOptMethods(cls);
}

} // namespace pyrol_flow_opt

PYBIND11_MODULE(_flow_opt_filtered_darcy, m) {
  m.doc() =
      "PyROL bindings for the axisymmetric filtered Darcy reduced objective";
  pyrol_flow_opt::bindFilteredDarcy(m);
}
