#include "PyROL_FlowOpt_Common.hpp"

#include "example/PDE-OPT/flow-opt/axisymmetric/models/brinkman/obj_brinkman.hpp"
#include "example/PDE-OPT/flow-opt/axisymmetric/models/brinkman/pde_brinkman.hpp"

namespace pyrol_flow_opt {

namespace {

class BrinkmanObjective final : public FlowOptObjectiveBase {
public:
  explicit BrinkmanObjective(const std::string &xmlPath)
      : FlowOptObjectiveBase(xmlPath) {
    finishBuild();
  }

protected:
  void buildModel() override {
    pde_ = ROL::makePtr<PDE_NavierStokes<RealT>>(*parlist_);
    con_ = ROL::makePtr<PDE_Constraint<RealT>>(pde_, meshMgr_, comm_, *parlist_,
                                               *outStream_);
    ROL::Ptr<PDE_Constraint<RealT>> pdecon =
        ROL::dynamicPtrCast<PDE_Constraint<RealT>>(con_);
    assembler_ = pdecon->getAssembler();
    con_->setSolveParameters(*parlist_);

    uField_ = assembler_->createStateVector();
    pField_ = assembler_->createStateVector();
    zField_ = assembler_->createControlVector();
    rField_ = assembler_->createResidualVector();
    uField_->randomize();
    pField_->randomize();
    zField_->randomize();
    rField_->putScalar(0.0);

    up_ = ROL::makePtr<PDE_PrimalSimVector<RealT>>(uField_, pde_, assembler_,
                                                   *parlist_);
    pp_ = ROL::makePtr<PDE_PrimalSimVector<RealT>>(pField_, pde_, assembler_,
                                                   *parlist_);
    zp_ = makeControlVector(zField_, pde_);
    rp_ = ROL::makePtr<PDE_DualSimVector<RealT>>(rField_, pde_, assembler_,
                                                 *parlist_);

    ROL::Ptr<PDE_NavierStokes<RealT>> brinkman =
        ROL::staticPtrCast<PDE_NavierStokes<RealT>>(pde_);
    ROL::Ptr<QoI<RealT>> qoi =
        ROL::makePtr<QoI_VelocityTracking_NavierStokes<RealT>>(
            *parlist_, brinkman->getVelocityFE(), brinkman->getPressureFE(),
            brinkman->getStateFieldInfo());
    ROL::Ptr<ROL::Objective_SimOpt<RealT>> obj =
        ROL::makePtr<PDE_Objective<RealT>>(qoi, assembler_);
    objective_ = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(
        obj, con_, up_, zp_, pp_, true, false);
  }
};

} // namespace

void bindBrinkman(py::module_ &m) {
  py::class_<BrinkmanObjective> cls(m, "_BrinkmanObjective");
  cls.def(py::init<const std::string &>(), py::arg("xml_path"));
  bindFlowOptMethods(cls);
}

} // namespace pyrol_flow_opt

PYBIND11_MODULE(_flow_opt_brinkman, m) {
  m.doc() = "PyROL bindings for the axisymmetric Brinkman reduced objective";
  pyrol_flow_opt::bindBrinkman(m);
}
