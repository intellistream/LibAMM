//
// Created by tony on 12/04/24.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <Utils/EigenTensor.h>
#include <Utils/ConfigMap.hpp>
#include <Utils/IntelliLog.h>
#include <LibAMM.h>
#include <include/papi_config.h>
#if LibAMM_PAPI == 1
#include <Utils/ThreadPerfPAPI.hpp>
#endif
namespace py = pybind11;
using namespace INTELLI;
using namespace LibAMM;
LibAMM::Tensor add_tensors(LibAMM::Tensor a, LibAMM::Tensor b) {
  return a + b;
}
py::dict configMapToDict(const std::shared_ptr<ConfigMap> &cfg) {
  py::dict d;
  auto i64Map = cfg->getI64Map();
  auto u64Map = cfg->getU64Map();
  auto doubleMap = cfg->getDoubleMap();
  auto strMap = cfg->getStrMap();
  for (auto &iter : i64Map) {
    d[py::cast(iter.first)] = py::cast(iter.second);
  }
  for (auto &iter : u64Map) {
    d[py::cast(iter.first)] = py::cast(iter.second);
  }
  for (auto &iter : doubleMap) {
    d[py::cast(iter.first)] = py::cast(iter.second);
  }
  for (auto &iter : strMap) {
    d[py::cast(iter.first)] = py::cast(iter.second);
  }
  return d;
}

// Function to convert Python dictionary to ConfigMap
std::shared_ptr<ConfigMap> dictToConfigMap(const py::dict &dict) {
  auto cfg = std::make_shared<ConfigMap>();
  for (auto item : dict) {
    auto key = py::str(item.first);
    auto value = item.second;
    // Check if the type is int
    if (py::isinstance<py::int_>(value)) {
      int64_t val = value.cast<int64_t>();
      cfg->edit(key, val);
      cfg->edit(key, (uint64_t) val);
     // std::cout << "Key: " << key.cast<std::string>() << " has an int value." << std::endl;
    }
      // Check if the type is float
    else if (py::isinstance<py::float_>(value)) {
      double val = value.cast<float>();
      cfg->edit(key, val);
    }
      // Check if the type is string
    else if (py::isinstance<py::str>(value)) {
      std::string val = py::str(value);
      cfg->edit(key, val);
    }
      // Add more type checks as needed
    else {
      std::cout << "Key: " << key.cast<std::string>() << " has a value of another type." << std::endl;
    }
  }
  return cfg;
}
AbstractCPPAlgoPtr createAMM(std::string nameTag) {
  CPPAlgoTable tab;
  auto ru = tab.findCppAlgo(nameTag);
  if (ru == nullptr) {
    INTELLI_ERROR("No algo named " + nameTag + ", return flat");
    nameTag = "mm";
    return tab.findCppAlgo(nameTag);
  }
  return ru;
}
AbstractMatrixLoaderPtr createMatrixLoader(std::string nameTag) {
  MatrixLoaderTable dt;
  auto ru = dt.findMatrixLoader(nameTag);
  if (ru == nullptr) {
    INTELLI_ERROR("No algo named " + nameTag + ", return flat");
    nameTag = "random";
    return dt.findMatrixLoader(nameTag);
  }
  return ru;
}

PYBIND11_MODULE(PyAMM, m) {
  /**
   * @brief export the configmap class
   */
  m.attr("__version__") = "0.1.0";  // Set the version of the module
  py::class_<INTELLI::ConfigMap, std::shared_ptr<INTELLI::ConfigMap>>(m, "ConfigMap")
      .def(py::init<>())
      .def("edit", py::overload_cast<const std::string &, int64_t>(&INTELLI::ConfigMap::edit))
      .def("edit", py::overload_cast<const std::string &, double>(&INTELLI::ConfigMap::edit))
      .def("edit", py::overload_cast<const std::string &, std::string>(&INTELLI::ConfigMap::edit))
      .def("toString", &INTELLI::ConfigMap::toString,
           py::arg("separator") = "\t",
           py::arg("newLine") = "\n")
      .def("toFile", &ConfigMap::toFile,
           py::arg("fname"),
           py::arg("separator") = ",",
           py::arg("newLine") = "\n")
      .def("fromFile", &ConfigMap::fromFile,
           py::arg("fname"),
           py::arg("separator") = ",",
           py::arg("newLine") = "\n");
  m.def("configMapToDict", &configMapToDict, "A function that converts ConfigMap to Python dictionary");
  m.def("dictToConfigMap", &dictToConfigMap, "A function that converts  Python dictionary to ConfigMap");
  m.def("createAMM", &createAMM, "A function to create new amm by name tag");
  m.def("createMatrixLoader", &createMatrixLoader, "A function to create new matrix loader by name tag");
  /**
   * @brief for CPP AMM algos
   */
  py::class_<AbstractCPPAlgo, std::shared_ptr<AbstractCPPAlgo>>(m, "AbstractCPPAlgo")
      .def(py::init<>())
      .def("setConfig", &AbstractCPPAlgo::setConfig)
      .def("amm", &AbstractCPPAlgo::amm)
      .def("getBreakDown", &AbstractCPPAlgo::getBreakDown);
  /**
   * @brief for matrix loaders
   */
  py::class_<AbstractMatrixLoader, std::shared_ptr<AbstractMatrixLoader>>(m, "AbstractMatrixLoader")
      .def(py::init<>())
      .def("setConfig", &AbstractMatrixLoader::setConfig)
      .def("getA", &AbstractMatrixLoader::getA)
      .def("getB", &AbstractMatrixLoader::getB);
  /***
   * @brief abstract index
   */
#if LibAMM_PAPI == 1
  py::class_<INTELLI::ThreadPerfPAPI, std::shared_ptr<INTELLI::ThreadPerfPAPI>>(m, "PAPIPerf")
      .def(py::init<>())
      .def("initEventsByCfg", &INTELLI::ThreadPerfPAPI::initEventsByCfg)
      .def("start", &INTELLI::ThreadPerfPAPI::start)
      .def("end", &INTELLI::ThreadPerfPAPI::end)
      .def("resultToConfigMap", &INTELLI::ThreadPerfPAPI::resultToConfigMap);
#endif
}