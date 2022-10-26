/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <qnnpack/params.h>

class LUTNormMicrokernelTester {
 public:
  inline LUTNormMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline LUTNormMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline LUTNormMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_u8lut32norm_ukernel_function u8lut32norm) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
    auto u32rng = std::bind(
        std::uniform_int_distribution<uint32_t>(
            1, std::numeric_limits<uint32_t>::max() / (257 * n())),
        rng);

    std::vector<uint8_t> x(n());
    std::vector<uint32_t> t(256);
    std::vector<uint8_t> y(n());
    std::vector<float> yRef(n());
    for(const auto iteration : c10::irange(iterations())) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::generate(t.begin(), t.end(), std::ref(u32rng));
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(u8rng));
      } else {
        std::fill(y.begin(), y.end(), 0xA5);
      }
      const uint8_t* xData = inplace() ? y.data() : x.data();

      /* Compute reference results */
      uint32_t sum = 0;
      for(const auto i : c10::irange(n())) {
        sum += t[xData[i]];
      }
      for(const auto i : c10::irange(n())) {
        yRef[i] = 256.0f * float(t[xData[i]]) / float(sum);
        yRef[i] = std::min(yRef[i], 255.0f);
      }

      /* Call optimized micro-kernel */
      u8lut32norm(n(), xData, t.data(), y.data());

      /* Verify results */
      for(const auto i : c10::irange(n())) {
        ASSERT_NEAR(yRef[i], float(y[i]), 0.5f)
            << "at position " << i << ", n = " << n() << ", sum = " << sum;
      }
    }
  }

 private:
  size_t n_{1};
  bool inplace_{false};
  size_t iterations_{15};
};
