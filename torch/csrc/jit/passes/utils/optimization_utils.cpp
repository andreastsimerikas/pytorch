#include <torch/csrc/jit/passes/utils/optimization_utils.h>

namespace torch {
namespace jit {

bool nonConstantParameters(Node* n) {
  // Checks if the parameters, not including the
  // first param are all constants.
  for(const auto i : c10::irange(1, n->inputs().size())) {
    if (n->inputs().at(i)->node()->kind() != prim::Constant) {
      return true;
    }
  }
  return false;
}

} // namespace jit
} // namespace torch
