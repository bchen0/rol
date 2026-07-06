import importlib

__version__ = '0.1.0'


def version():
    return 'PyROL version: ' + __version__


try:
    from . getTypeName import getTypeName, getDefaultScalarType, ROL_classes, ROL_members
    from pyrol.pyrol.Teuchos import ParameterList
except ModuleNotFoundError as exc:
    if exc.name != 'pyrol.pyrol':
        raise
else:
    def getWrapper(classname):
        def wrapper(scalarType=getDefaultScalarType()):
            actualClass = getTypeName(classname, scalarType)
            return actualClass
        return wrapper


    defaultScalarType = getDefaultScalarType()

    supported_objects = {"BoundConstraint",
                         "Bounds",
                         "Constraint",
                         "getCout",
                         "getParametersFromXmlFile",
                         "getParametersFromYamlFile",
                         "LinearConstraint",
                         "LinearOperator",
                         "Objective",
                         "Problem",
                         "Solver",
                         "Vector"}

    unsupported = importlib.import_module('.unsupported', 'pyrol')

    for classnameLong in ROL_members:
        class_obj, _ = ROL_members[classnameLong]
        pos = classnameLong.find('_'+defaultScalarType+'_t')
        if pos <= 0:
            if classnameLong in supported_objects:
                locals().update({classnameLong: class_obj})
            else:
                setattr(unsupported, classnameLong, class_obj)
            continue

        classname = classnameLong[:pos]

        if classname in supported_objects:
            locals().update({classname: getTypeName(classname, defaultScalarType)})
            locals().update({classname+'_forScalar': getWrapper(classname)})
        else:
            setattr(unsupported, classname, getTypeName(classname, defaultScalarType))
            setattr(unsupported, classname + '_forScalar', getWrapper(classname))

    del getTypeName, getDefaultScalarType, ROL_classes, ROL_members
    del class_obj, classname, classnameLong, getWrapper, pos

del importlib
