// Minimal Amesos2/KLU2 sparse solve helper for the Python Navier-Stokes port.
//
// Input format:
//   n nnz has_global_ids
//   gid_0 ... gid_{n-1}        (only when has_global_ids is 1)
//   row col value
//   ... nnz entries ...
//   rhs_0
//   ... rhs_{n-1}
//
// The rows and columns are zero-based and serial.  The program prints:
//   {"solution":[...]}

#include "Amesos2.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include <cstddef>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct SparseProblem {
  std::size_t n = 0;
  std::vector<long long> globalIds;
  std::vector<std::vector<int>> cols;
  std::vector<std::vector<double>> vals;
  std::vector<double> rhs;
};

SparseProblem readProblem(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Could not open sparse problem file: " + path);
  }

  SparseProblem problem;
  std::size_t nnz = 0;
  int hasGlobalIds = 0;
  input >> problem.n >> nnz >> hasGlobalIds;
  if (!input) {
    throw std::runtime_error("Could not read sparse problem header");
  }

  problem.globalIds.resize(problem.n);
  if (hasGlobalIds) {
    for (std::size_t i = 0; i < problem.n; ++i) {
      input >> problem.globalIds[i];
      if (!input) {
        throw std::runtime_error("Could not read global ID entry");
      }
    }
  } else {
    for (std::size_t i = 0; i < problem.n; ++i) {
      problem.globalIds[i] = static_cast<long long>(i);
    }
  }

  problem.cols.assign(problem.n, {});
  problem.vals.assign(problem.n, {});
  for (std::size_t k = 0; k < nnz; ++k) {
    std::size_t row = 0;
    int col = 0;
    double val = 0.0;
    input >> row >> col >> val;
    if (!input || row >= problem.n || col < 0 || static_cast<std::size_t>(col) >= problem.n) {
      throw std::runtime_error("Invalid sparse matrix entry");
    }
    problem.cols[row].push_back(col);
    problem.vals[row].push_back(val);
  }

  problem.rhs.resize(problem.n);
  for (std::size_t i = 0; i < problem.n; ++i) {
    input >> problem.rhs[i];
    if (!input) {
      throw std::runtime_error("Could not read RHS entry");
    }
  }
  return problem;
}

void printSolution(const Tpetra::MultiVector<> &x, const std::size_t n) {
  const Teuchos::ArrayRCP<const double> data = x.getData(0);
  std::cout << std::scientific << std::setprecision(17);
  std::cout << "{\"solution\":[";
  for (std::size_t i = 0; i < n; ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << data[static_cast<int>(i)];
  }
  std::cout << "]}" << std::endl;
}

} // namespace

int main(int argc, char *argv[]) {
  bool transpose = false;
  std::string inputPath;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--transpose") {
      transpose = true;
    } else if (inputPath.empty()) {
      inputPath = arg;
    } else {
      throw std::runtime_error("Usage: klu2_sparse_solve.exe problem.txt [--transpose]");
    }
  }
  if (inputPath.empty()) {
    throw std::runtime_error("Usage: klu2_sparse_solve.exe problem.txt [--transpose]");
  }

  Tpetra::ScopeGuard tpetraScope(&argc, &argv);
  auto comm = Tpetra::getDefaultComm();
  if (comm->getSize() != 1) {
    throw std::runtime_error("klu2_sparse_solve expects a serial communicator");
  }

  using matrix_type = Tpetra::CrsMatrix<>;
  using vector_type = Tpetra::MultiVector<>;
  using map_type = Tpetra::Map<>;
  using global_ordinal_type = matrix_type::global_ordinal_type;
  const SparseProblem problem = readProblem(inputPath);

  const Tpetra::global_size_t invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  std::vector<global_ordinal_type> globalIds(problem.globalIds.begin(), problem.globalIds.end());
  const Teuchos::ArrayView<const global_ordinal_type> globalIdView(globalIds.data(), globalIds.size());
  auto map = Teuchos::rcp(new map_type(invalid, globalIdView, static_cast<global_ordinal_type>(0), comm));
  std::size_t maxEntriesPerRow = 0;
  for (const auto &rowCols : problem.cols) {
    maxEntriesPerRow = std::max(maxEntriesPerRow, rowCols.size());
  }
  auto matrix = Teuchos::rcp(new matrix_type(map, maxEntriesPerRow));
  for (std::size_t row = 0; row < problem.n; ++row) {
    const auto &rowColsInt = problem.cols[row];
    const auto &rowVals = problem.vals[row];
    std::vector<global_ordinal_type> rowCols;
    rowCols.reserve(rowColsInt.size());
    for (const int col : rowColsInt) {
      rowCols.push_back(globalIds[static_cast<std::size_t>(col)]);
    }
    const Teuchos::ArrayView<const global_ordinal_type> colView(rowCols.data(), rowCols.size());
    const Teuchos::ArrayView<const double> valView(rowVals.data(), rowVals.size());
    matrix->insertGlobalValues(globalIds[row], colView, valView);
  }
  matrix->fillComplete();

  auto rhs = Teuchos::rcp(new vector_type(map, 1));
  auto lhs = Teuchos::rcp(new vector_type(map, 1));
  rhs->putScalar(0.0);
  lhs->putScalar(0.0);
  for (std::size_t i = 0; i < problem.n; ++i) {
    rhs->replaceGlobalValue(globalIds[i], 0, problem.rhs[i]);
  }

  auto solver = Amesos2::create<matrix_type, vector_type>("KLU2", matrix);
  solver->symbolicFactorization();
  solver->numericFactorization();

  auto params = Teuchos::rcp(new Teuchos::ParameterList("Amesos2"));
  params->sublist("KLU2").set("Transpose", transpose);
  solver->setParameters(params);
  solver->setX(lhs);
  solver->setB(rhs);
  solver->solve();

  printSolution(*lhs, problem.n);
  return 0;
}
