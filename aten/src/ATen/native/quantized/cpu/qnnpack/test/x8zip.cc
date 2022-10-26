/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include <qnnpack/isa-checks.h>
#include <qnnpack/x8zip.h>

#include "zip-microkernel-tester.h"

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
TEST(X8ZIP_X2__NEON, n_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  ZipMicrokernelTester().n(8).g(2).test(pytorch_qnnp_x8zip_x2__neon);
}

TEST(X8ZIP_X2__NEON, n_div_16) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 8) {
    ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__neon);
  }
}

TEST(X8ZIP_X2__NEON, n_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for(const auto n : c10::irange(9, 16)) {
    ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__neon);
  }
}

TEST(X8ZIP_X2__NEON, n_lt_8) {
  TEST_REQUIRES_ARM_NEON;
  for(const auto n : c10::irange(1, 8)) {
    ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__neon);
  }
}

TEST(X8ZIP_X3__NEON, n_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  ZipMicrokernelTester().n(9).g(3).test(pytorch_qnnp_x8zip_x3__neon);
}

TEST(X8ZIP_X3__NEON, n_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 8) {
    ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__neon);
  }
}

TEST(X8ZIP_X3__NEON, n_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for(const auto n : c10::irange(9, 16)) {
    ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__neon);
  }
}

TEST(X8ZIP_X3__NEON, n_lt_8) {
  TEST_REQUIRES_ARM_NEON;
  for(const auto n : c10::irange(1, 8)) {
    ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__neon);
  }
}

TEST(X8ZIP_X4__NEON, n_eq_8) {
  TEST_REQUIRES_ARM_NEON;
  ZipMicrokernelTester().n(8).g(4).test(pytorch_qnnp_x8zip_x4__neon);
}

TEST(X8ZIP_X4__NEON, n_div_8) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 8) {
    ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__neon);
  }
}

TEST(X8ZIP_X4__NEON, n_gt_8) {
  TEST_REQUIRES_ARM_NEON;
  for(const auto n : c10::irange(9, 16)) {
    ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__neon);
  }
}

TEST(X8ZIP_X4__NEON, n_lt_16) {
  TEST_REQUIRES_ARM_NEON;
  for(const auto n : c10::irange(1, 16)) {
    ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__neon);
  }
}

TEST(X8ZIP_XM__NEON, n_eq_8_m_eq_4) {
  TEST_REQUIRES_ARM_NEON;
  ZipMicrokernelTester().n(8).g(4).test(pytorch_qnnp_x8zip_xm__neon);
}

TEST(X8ZIP_XM__NEON, n_eq_8_m_div_4) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t g = 4; g < 32; g += 4) {
    ZipMicrokernelTester().n(8).g(g).test(pytorch_qnnp_x8zip_xm__neon);
  }
}

TEST(X8ZIP_XM__NEON, n_eq_8_m_gt_4) {
  TEST_REQUIRES_ARM_NEON;
  for(const auto g : c10::irange(5, 8)) {
    ZipMicrokernelTester().n(8).g(g).test(pytorch_qnnp_x8zip_xm__neon);
  }
}

TEST(X8ZIP_XM__NEON, n_div_8_m_eq_4) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 8) {
    ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_xm__neon);
  }
}

TEST(X8ZIP_XM__NEON, n_div_8_m_div_4) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 8) {
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__neon);
    }
  }
}

TEST(X8ZIP_XM__NEON, n_div_8_m_gt_4) {
  TEST_REQUIRES_ARM_NEON;
  for (size_t n = 8; n < 128; n += 8) {
    for(const auto g : c10::irange(5, 8)) {
      ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__neon);
    }
  }
}

TEST(X8ZIP_XM__NEON, n_gt_8_m_eq_4) {
  TEST_REQUIRES_ARM_NEON;
  for(const auto n : c10::irange(9, 16)) {
    ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_xm__neon);
  }
}

TEST(X8ZIP_XM__NEON, n_gt_8_m_div_4) {
  TEST_REQUIRES_ARM_NEON;
  for(const auto n : c10::irange(9, 16)) {
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__neon);
    }
  }
}

TEST(X8ZIP_XM__NEON, n_gt_8_m_gt_4) {
  TEST_REQUIRES_ARM_NEON;
  for(const auto n : c10::irange(9, 16)) {
    for(const auto g : c10::irange(5, 8)) {
      ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__neon);
    }
  }
}

TEST(X8ZIP_XM__NEON, n_lt_8) {
  TEST_REQUIRES_ARM_NEON;
  for(const auto n : c10::irange(1, 8)) {
    for(const auto g : c10::irange(4, 12)) {
      ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__neon);
    }
  }
}
#endif

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
TEST(X8ZIP_X2__SSE2, n_eq_16) {
  TEST_REQUIRES_X86_SSE2;
  ZipMicrokernelTester().n(16).g(2).test(pytorch_qnnp_x8zip_x2__sse2);
}

TEST(X8ZIP_X2__SSE2, n_div_16) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 16; n < 256; n += 16) {
    ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__sse2);
  }
}

TEST(X8ZIP_X2__SSE2, n_gt_16) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto n : c10::irange(17, 32)) {
    ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__sse2);
  }
}

TEST(X8ZIP_X2__SSE2, n_lt_16) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto n : c10::irange(1, 16)) {
    ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__sse2);
  }
}

TEST(X8ZIP_X3__SSE2, n_eq_16) {
  TEST_REQUIRES_X86_SSE2;
  ZipMicrokernelTester().n(16).g(3).test(pytorch_qnnp_x8zip_x3__sse2);
}

TEST(X8ZIP_X3__SSE2, n_div_16) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 16; n < 256; n += 16) {
    ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__sse2);
  }
}

TEST(X8ZIP_X3__SSE2, n_gt_16) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto n : c10::irange(17, 32)) {
    ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__sse2);
  }
}

TEST(X8ZIP_X3__SSE2, n_lt_16) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto n : c10::irange(1, 16)) {
    ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__sse2);
  }
}

TEST(X8ZIP_X4__SSE2, n_eq_16) {
  TEST_REQUIRES_X86_SSE2;
  ZipMicrokernelTester().n(16).g(4).test(pytorch_qnnp_x8zip_x4__sse2);
}

TEST(X8ZIP_X4__SSE2, n_div_16) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 16; n < 256; n += 16) {
    ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__sse2);
  }
}

TEST(X8ZIP_X4__SSE2, n_gt_16) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto n : c10::irange(17, 32)) {
    ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__sse2);
  }
}

TEST(X8ZIP_X4__SSE2, n_lt_16) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto n : c10::irange(1, 16)) {
    ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__sse2);
  }
}

TEST(X8ZIP_XM__SSE2, n_eq_8_m_eq_4) {
  TEST_REQUIRES_X86_SSE2;
  ZipMicrokernelTester().n(8).g(4).test(pytorch_qnnp_x8zip_xm__sse2);
}

TEST(X8ZIP_XM__SSE2, n_eq_8_m_div_4) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t g = 4; g < 32; g += 4) {
    ZipMicrokernelTester().n(8).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
  }
}

TEST(X8ZIP_XM__SSE2, n_eq_8_m_gt_4) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto g : c10::irange(5, 8)) {
    ZipMicrokernelTester().n(8).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
  }
}

TEST(X8ZIP_XM__SSE2, n_eq_16_m_eq_4) {
  TEST_REQUIRES_X86_SSE2;
  ZipMicrokernelTester().n(16).g(4).test(pytorch_qnnp_x8zip_xm__sse2);
}

TEST(X8ZIP_XM__SSE2, n_eq_16_m_div_4) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t g = 4; g < 32; g += 4) {
    ZipMicrokernelTester().n(16).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
  }
}

TEST(X8ZIP_XM__SSE2, n_eq_16_m_gt_4) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto g : c10::irange(5, 8)) {
    ZipMicrokernelTester().n(16).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
  }
}

TEST(X8ZIP_XM__SSE2, n_div_16_m_eq_4) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 16; n < 256; n += 16) {
    ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_xm__sse2);
  }
}

TEST(X8ZIP_XM__SSE2, n_div_16_m_div_4) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 16; n < 256; n += 16) {
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
    }
  }
}

TEST(X8ZIP_XM__SSE2, n_div_16_m_gt_4) {
  TEST_REQUIRES_X86_SSE2;
  for (size_t n = 16; n < 256; n += 16) {
    for(const auto g : c10::irange(5, 8)) {
      ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
    }
  }
}

TEST(X8ZIP_XM__SSE2, n_gt_16_m_eq_4) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto n : c10::irange(17, 32)) {
    ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_xm__sse2);
  }
}

TEST(X8ZIP_XM__SSE2, n_gt_16_m_div_4) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto n : c10::irange(17, 32)) {
    for (size_t g = 4; g < 32; g += 4) {
      ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
    }
  }
}

TEST(X8ZIP_XM__SSE2, n_gt_16_m_gt_4) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto n : c10::irange(17, 32)) {
    for(const auto g : c10::irange(5, 8)) {
      ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
    }
  }
}

TEST(X8ZIP_XM__SSE2, n_lt_16) {
  TEST_REQUIRES_X86_SSE2;
  for(const auto n : c10::irange(1, 16)) {
    for(const auto g : c10::irange(4, 12)) {
      ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
    }
  }
}
#endif
