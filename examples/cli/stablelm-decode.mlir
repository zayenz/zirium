// Adapted from MLXcel's StableLM decode example at commit 0accedd90ae9ea0679121bd087dafdd82882182a.
// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin. Licensed under Apache-2.0.
// This product includes software developed at Lablup Inc.
// Zirium replaces the embedded rotary tables with zero splats to keep this example readable.
// Source: https://github.com/lablup/mlxcel/blob/0accedd90ae9ea0679121bd087dafdd82882182a/src/lib/mlxcel-xla/assets/stablelm/decode.mlir

module @decode_step {
  func.func public @main(%arg0: tensor<32x32xf32> loc("params['embed']"), %arg1: tensor<32xf32> loc("params['final_norm']"), %arg2: tensor<32xf32> loc("params['final_norm_bias']"), %arg3: tensor<32x32xf32> loc("params['lm_head']"), %arg4: tensor<32x64xf32> loc("params['layers'][0]['down']"), %arg5: tensor<64x32xf32> loc("params['layers'][0]['gate']"), %arg6: tensor<32xf32> loc("params['layers'][0]['in_ln']"), %arg7: tensor<32xf32> loc("params['layers'][0]['post_ln']"), %arg8: tensor<64x32xf32> loc("params['layers'][0]['up']"), %arg9: tensor<16x32xf32> loc("params['layers'][0]['wk']"), %arg10: tensor<32x32xf32> loc("params['layers'][0]['wo']"), %arg11: tensor<32x32xf32> loc("params['layers'][0]['wq']"), %arg12: tensor<16x32xf32> loc("params['layers'][0]['wv']"), %arg13: tensor<16xf32> loc("params['layers'][0]['bk']"), %arg14: tensor<32xf32> loc("params['layers'][0]['bq']"), %arg15: tensor<16xf32> loc("params['layers'][0]['bv']"), %arg16: tensor<32xf32> loc("params['layers'][0]['in_ln_bias']"), %arg17: tensor<32xf32> loc("params['layers'][0]['post_ln_bias']"), %arg18: tensor<32x64xf32> loc("params['layers'][1]['down']"), %arg19: tensor<64x32xf32> loc("params['layers'][1]['gate']"), %arg20: tensor<32xf32> loc("params['layers'][1]['in_ln']"), %arg21: tensor<32xf32> loc("params['layers'][1]['post_ln']"), %arg22: tensor<64x32xf32> loc("params['layers'][1]['up']"), %arg23: tensor<16x32xf32> loc("params['layers'][1]['wk']"), %arg24: tensor<32x32xf32> loc("params['layers'][1]['wo']"), %arg25: tensor<32x32xf32> loc("params['layers'][1]['wq']"), %arg26: tensor<16x32xf32> loc("params['layers'][1]['wv']"), %arg27: tensor<16xf32> loc("params['layers'][1]['bk']"), %arg28: tensor<32xf32> loc("params['layers'][1]['bq']"), %arg29: tensor<16xf32> loc("params['layers'][1]['bv']"), %arg30: tensor<32xf32> loc("params['layers'][1]['in_ln_bias']"), %arg31: tensor<32xf32> loc("params['layers'][1]['post_ln_bias']"), %arg32: tensor<i32> loc("token"), %arg33: tensor<i32> loc("pos"), %arg34: tensor<i32> loc("cache_len"), %arg35: tensor<2x256x2x8xf32> loc("kcache"), %arg36: tensor<2x256x2x8xf32> loc("vcache")) -> (tensor<32xf32>, tensor<2x256x2x8xf32>, tensor<2x256x2x8xf32>) {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<256x2xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<256x2xf32>
    %2 = stablehlo.constant dense<0x00000000> : tensor<f32>
    %3 = stablehlo.constant dense<0x3F800000> : tensor<f32>
    %4 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.constant dense<0xF149F2CA> : tensor<f32>
    %6 = stablehlo.constant dense<0x3727C5AC> : tensor<f32>
    %7 = stablehlo.constant dense<0x42000000> : tensor<f32>
    %8 = stablehlo.constant dense<0x3EB504F3> : tensor<f32>
    %9 = stablehlo.constant dense<0> : tensor<i32>
    %10 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.constant dense<1> : tensor<i32>
    %12 = stablehlo.dynamic_slice %arg0, %arg32, %9, sizes = [1, 32] : (tensor<32x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x32xf32>
    %13 = stablehlo.reshape %12 : (tensor<1x32xf32>) -> tensor<32xf32>
    %14 = stablehlo.dynamic_slice %0, %arg33, %9, sizes = [1, 2] : (tensor<256x2xf32>, tensor<i32>, tensor<i32>) -> tensor<1x2xf32>
    %15 = stablehlo.reshape %14 : (tensor<1x2xf32>) -> tensor<2xf32>
    %16 = stablehlo.dynamic_slice %1, %arg33, %9, sizes = [1, 2] : (tensor<256x2xf32>, tensor<i32>, tensor<i32>) -> tensor<1x2xf32>
    %17 = stablehlo.reshape %16 : (tensor<1x2xf32>) -> tensor<2xf32>
    %18 = stablehlo.iota dim = 0 : tensor<256xi32>
    %19 = stablehlo.broadcast_in_dim %arg34, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %20 = stablehlo.compare LE, %18, %19, SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %21 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %22 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %23 = stablehlo.select %20, %21, %22 : tensor<256xi1>, tensor<256xf32>
    %24 = stablehlo.reduce(%13 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %25 = stablehlo.divide %24, %7 : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %27 = stablehlo.subtract %13, %26 : tensor<32xf32>
    %28 = stablehlo.multiply %27, %27 : tensor<32xf32>
    %29 = stablehlo.reduce(%28 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %30 = stablehlo.divide %29, %7 : tensor<f32>
    %31 = stablehlo.add %30, %6 : tensor<f32>
    %32 = stablehlo.rsqrt %31 : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %34 = stablehlo.multiply %27, %33 : tensor<32xf32>
    %35 = stablehlo.multiply %34, %arg6 : tensor<32xf32>
    %36 = stablehlo.add %35, %arg16 : tensor<32xf32>
    %37 = stablehlo.dot_general %36, %arg11, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<32x32xf32>) -> tensor<32xf32>
    %38 = stablehlo.add %37, %arg14 : tensor<32xf32>
    %39 = stablehlo.reshape %38 : (tensor<32xf32>) -> tensor<4x8xf32>
    %40 = stablehlo.dot_general %36, %arg9, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<16x32xf32>) -> tensor<16xf32>
    %41 = stablehlo.add %40, %arg13 : tensor<16xf32>
    %42 = stablehlo.reshape %41 : (tensor<16xf32>) -> tensor<2x8xf32>
    %43 = stablehlo.dot_general %36, %arg12, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<16x32xf32>) -> tensor<16xf32>
    %44 = stablehlo.add %43, %arg15 : tensor<16xf32>
    %45 = stablehlo.reshape %44 : (tensor<16xf32>) -> tensor<2x8xf32>
    %46 = stablehlo.slice %39 [0:4, 0:2] : (tensor<4x8xf32>) -> tensor<4x2xf32>
    %47 = stablehlo.slice %39 [0:4, 2:8] : (tensor<4x8xf32>) -> tensor<4x6xf32>
    %48 = stablehlo.broadcast_in_dim %15, dims = [1] : (tensor<2xf32>) -> tensor<4x2xf32>
    %49 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<2xf32>) -> tensor<4x2xf32>
    %50 = stablehlo.multiply %46, %48 : tensor<4x2xf32>
    %51 = stablehlo.slice %46 [0:4, 0:1] : (tensor<4x2xf32>) -> tensor<4x1xf32>
    %52 = stablehlo.slice %46 [0:4, 1:2] : (tensor<4x2xf32>) -> tensor<4x1xf32>
    %53 = stablehlo.negate %52 : tensor<4x1xf32>
    %54 = stablehlo.concatenate %53, %51, dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
    %55 = stablehlo.multiply %54, %49 : tensor<4x2xf32>
    %56 = stablehlo.add %50, %55 : tensor<4x2xf32>
    %57 = stablehlo.concatenate %56, %47, dim = 1 : (tensor<4x2xf32>, tensor<4x6xf32>) -> tensor<4x8xf32>
    %58 = stablehlo.slice %42 [0:2, 0:2] : (tensor<2x8xf32>) -> tensor<2x2xf32>
    %59 = stablehlo.slice %42 [0:2, 2:8] : (tensor<2x8xf32>) -> tensor<2x6xf32>
    %60 = stablehlo.broadcast_in_dim %15, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %61 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %62 = stablehlo.multiply %58, %60 : tensor<2x2xf32>
    %63 = stablehlo.slice %58 [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %64 = stablehlo.slice %58 [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %65 = stablehlo.negate %64 : tensor<2x1xf32>
    %66 = stablehlo.concatenate %65, %63, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
    %67 = stablehlo.multiply %66, %61 : tensor<2x2xf32>
    %68 = stablehlo.add %62, %67 : tensor<2x2xf32>
    %69 = stablehlo.concatenate %68, %59, dim = 1 : (tensor<2x2xf32>, tensor<2x6xf32>) -> tensor<2x8xf32>
    %70 = stablehlo.reshape %69 : (tensor<2x8xf32>) -> tensor<1x1x2x8xf32>
    %71 = stablehlo.dynamic_update_slice %arg35, %70, %10, %arg34, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x1x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %72 = stablehlo.reshape %45 : (tensor<2x8xf32>) -> tensor<1x1x2x8xf32>
    %73 = stablehlo.dynamic_update_slice %arg36, %72, %10, %arg34, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x1x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %74 = stablehlo.slice %71 [0:1, 0:256, 0:2, 0:8] : (tensor<2x256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %75 = stablehlo.reshape %74 : (tensor<1x256x2x8xf32>) -> tensor<256x2x8xf32>
    %76 = stablehlo.slice %73 [0:1, 0:256, 0:2, 0:8] : (tensor<2x256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %77 = stablehlo.reshape %76 : (tensor<1x256x2x8xf32>) -> tensor<256x2x8xf32>
    %78 = stablehlo.reshape %57 : (tensor<4x8xf32>) -> tensor<2x2x8xf32>
    %79 = stablehlo.dot_general %78, %75, batching_dims = [0] x [1], contracting_dims = [2] x [2] : (tensor<2x2x8xf32>, tensor<256x2x8xf32>) -> tensor<2x2x256xf32>
    %80 = stablehlo.reshape %79 : (tensor<2x2x256xf32>) -> tensor<4x256xf32>
    %81 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4x256xf32>
    %82 = stablehlo.multiply %80, %81 : tensor<4x256xf32>
    %83 = stablehlo.broadcast_in_dim %23, dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
    %84 = stablehlo.add %82, %83 : tensor<4x256xf32>
    %85 = stablehlo.reduce(%84 init: %4) applies stablehlo.maximum across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %86 = stablehlo.broadcast_in_dim %85, dims = [0] : (tensor<4xf32>) -> tensor<4x256xf32>
    %87 = stablehlo.subtract %84, %86 : tensor<4x256xf32>
    %88 = stablehlo.exponential %87 : tensor<4x256xf32>
    %89 = stablehlo.reduce(%88 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [0] : (tensor<4xf32>) -> tensor<4x256xf32>
    %91 = stablehlo.divide %88, %90 : tensor<4x256xf32>
    %92 = stablehlo.reshape %91 : (tensor<4x256xf32>) -> tensor<2x2x256xf32>
    %93 = stablehlo.dot_general %92, %77, batching_dims = [0] x [1], contracting_dims = [2] x [0] : (tensor<2x2x256xf32>, tensor<256x2x8xf32>) -> tensor<2x2x8xf32>
    %94 = stablehlo.reshape %93 : (tensor<2x2x8xf32>) -> tensor<4x8xf32>
    %95 = stablehlo.reshape %94 : (tensor<4x8xf32>) -> tensor<32xf32>
    %96 = stablehlo.dot_general %95, %arg10, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<32x32xf32>) -> tensor<32xf32>
    %97 = stablehlo.add %13, %96 : tensor<32xf32>
    %98 = stablehlo.reduce(%97 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %99 = stablehlo.divide %98, %7 : tensor<f32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %101 = stablehlo.subtract %97, %100 : tensor<32xf32>
    %102 = stablehlo.multiply %101, %101 : tensor<32xf32>
    %103 = stablehlo.reduce(%102 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %104 = stablehlo.divide %103, %7 : tensor<f32>
    %105 = stablehlo.add %104, %6 : tensor<f32>
    %106 = stablehlo.rsqrt %105 : tensor<f32>
    %107 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %108 = stablehlo.multiply %101, %107 : tensor<32xf32>
    %109 = stablehlo.multiply %108, %arg7 : tensor<32xf32>
    %110 = stablehlo.add %109, %arg17 : tensor<32xf32>
    %111 = stablehlo.dot_general %110, %arg5, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<64x32xf32>) -> tensor<64xf32>
    %112 = stablehlo.dot_general %110, %arg8, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<64x32xf32>) -> tensor<64xf32>
    %113 = stablehlo.negate %111 : tensor<64xf32>
    %114 = stablehlo.exponential %113 : tensor<64xf32>
    %115 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %116 = stablehlo.add %115, %114 : tensor<64xf32>
    %117 = stablehlo.divide %115, %116 : tensor<64xf32>
    %118 = stablehlo.multiply %111, %117 : tensor<64xf32>
    %119 = stablehlo.multiply %118, %112 : tensor<64xf32>
    %120 = stablehlo.dot_general %119, %arg4, contracting_dims = [0] x [1] : (tensor<64xf32>, tensor<32x64xf32>) -> tensor<32xf32>
    %121 = stablehlo.add %97, %120 : tensor<32xf32>
    %122 = stablehlo.reduce(%121 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %123 = stablehlo.divide %122, %7 : tensor<f32>
    %124 = stablehlo.broadcast_in_dim %123, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %125 = stablehlo.subtract %121, %124 : tensor<32xf32>
    %126 = stablehlo.multiply %125, %125 : tensor<32xf32>
    %127 = stablehlo.reduce(%126 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %128 = stablehlo.divide %127, %7 : tensor<f32>
    %129 = stablehlo.add %128, %6 : tensor<f32>
    %130 = stablehlo.rsqrt %129 : tensor<f32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %132 = stablehlo.multiply %125, %131 : tensor<32xf32>
    %133 = stablehlo.multiply %132, %arg20 : tensor<32xf32>
    %134 = stablehlo.add %133, %arg30 : tensor<32xf32>
    %135 = stablehlo.dot_general %134, %arg25, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<32x32xf32>) -> tensor<32xf32>
    %136 = stablehlo.add %135, %arg28 : tensor<32xf32>
    %137 = stablehlo.reshape %136 : (tensor<32xf32>) -> tensor<4x8xf32>
    %138 = stablehlo.dot_general %134, %arg23, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<16x32xf32>) -> tensor<16xf32>
    %139 = stablehlo.add %138, %arg27 : tensor<16xf32>
    %140 = stablehlo.reshape %139 : (tensor<16xf32>) -> tensor<2x8xf32>
    %141 = stablehlo.dot_general %134, %arg26, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<16x32xf32>) -> tensor<16xf32>
    %142 = stablehlo.add %141, %arg29 : tensor<16xf32>
    %143 = stablehlo.reshape %142 : (tensor<16xf32>) -> tensor<2x8xf32>
    %144 = stablehlo.slice %137 [0:4, 0:2] : (tensor<4x8xf32>) -> tensor<4x2xf32>
    %145 = stablehlo.slice %137 [0:4, 2:8] : (tensor<4x8xf32>) -> tensor<4x6xf32>
    %146 = stablehlo.broadcast_in_dim %15, dims = [1] : (tensor<2xf32>) -> tensor<4x2xf32>
    %147 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<2xf32>) -> tensor<4x2xf32>
    %148 = stablehlo.multiply %144, %146 : tensor<4x2xf32>
    %149 = stablehlo.slice %144 [0:4, 0:1] : (tensor<4x2xf32>) -> tensor<4x1xf32>
    %150 = stablehlo.slice %144 [0:4, 1:2] : (tensor<4x2xf32>) -> tensor<4x1xf32>
    %151 = stablehlo.negate %150 : tensor<4x1xf32>
    %152 = stablehlo.concatenate %151, %149, dim = 1 : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<4x2xf32>
    %153 = stablehlo.multiply %152, %147 : tensor<4x2xf32>
    %154 = stablehlo.add %148, %153 : tensor<4x2xf32>
    %155 = stablehlo.concatenate %154, %145, dim = 1 : (tensor<4x2xf32>, tensor<4x6xf32>) -> tensor<4x8xf32>
    %156 = stablehlo.slice %140 [0:2, 0:2] : (tensor<2x8xf32>) -> tensor<2x2xf32>
    %157 = stablehlo.slice %140 [0:2, 2:8] : (tensor<2x8xf32>) -> tensor<2x6xf32>
    %158 = stablehlo.broadcast_in_dim %15, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %159 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %160 = stablehlo.multiply %156, %158 : tensor<2x2xf32>
    %161 = stablehlo.slice %156 [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %162 = stablehlo.slice %156 [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %163 = stablehlo.negate %162 : tensor<2x1xf32>
    %164 = stablehlo.concatenate %163, %161, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
    %165 = stablehlo.multiply %164, %159 : tensor<2x2xf32>
    %166 = stablehlo.add %160, %165 : tensor<2x2xf32>
    %167 = stablehlo.concatenate %166, %157, dim = 1 : (tensor<2x2xf32>, tensor<2x6xf32>) -> tensor<2x8xf32>
    %168 = stablehlo.reshape %167 : (tensor<2x8xf32>) -> tensor<1x1x2x8xf32>
    %169 = stablehlo.dynamic_update_slice %71, %168, %11, %arg34, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x1x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %170 = stablehlo.reshape %143 : (tensor<2x8xf32>) -> tensor<1x1x2x8xf32>
    %171 = stablehlo.dynamic_update_slice %73, %170, %11, %arg34, %9, %9 : (tensor<2x256x2x8xf32>, tensor<1x1x2x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x256x2x8xf32>
    %172 = stablehlo.slice %169 [1:2, 0:256, 0:2, 0:8] : (tensor<2x256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %173 = stablehlo.reshape %172 : (tensor<1x256x2x8xf32>) -> tensor<256x2x8xf32>
    %174 = stablehlo.slice %171 [1:2, 0:256, 0:2, 0:8] : (tensor<2x256x2x8xf32>) -> tensor<1x256x2x8xf32>
    %175 = stablehlo.reshape %174 : (tensor<1x256x2x8xf32>) -> tensor<256x2x8xf32>
    %176 = stablehlo.reshape %155 : (tensor<4x8xf32>) -> tensor<2x2x8xf32>
    %177 = stablehlo.dot_general %176, %173, batching_dims = [0] x [1], contracting_dims = [2] x [2] : (tensor<2x2x8xf32>, tensor<256x2x8xf32>) -> tensor<2x2x256xf32>
    %178 = stablehlo.reshape %177 : (tensor<2x2x256xf32>) -> tensor<4x256xf32>
    %179 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4x256xf32>
    %180 = stablehlo.multiply %178, %179 : tensor<4x256xf32>
    %181 = stablehlo.broadcast_in_dim %23, dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
    %182 = stablehlo.add %180, %181 : tensor<4x256xf32>
    %183 = stablehlo.reduce(%182 init: %4) applies stablehlo.maximum across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %184 = stablehlo.broadcast_in_dim %183, dims = [0] : (tensor<4xf32>) -> tensor<4x256xf32>
    %185 = stablehlo.subtract %182, %184 : tensor<4x256xf32>
    %186 = stablehlo.exponential %185 : tensor<4x256xf32>
    %187 = stablehlo.reduce(%186 init: %2) applies stablehlo.add across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0] : (tensor<4xf32>) -> tensor<4x256xf32>
    %189 = stablehlo.divide %186, %188 : tensor<4x256xf32>
    %190 = stablehlo.reshape %189 : (tensor<4x256xf32>) -> tensor<2x2x256xf32>
    %191 = stablehlo.dot_general %190, %175, batching_dims = [0] x [1], contracting_dims = [2] x [0] : (tensor<2x2x256xf32>, tensor<256x2x8xf32>) -> tensor<2x2x8xf32>
    %192 = stablehlo.reshape %191 : (tensor<2x2x8xf32>) -> tensor<4x8xf32>
    %193 = stablehlo.reshape %192 : (tensor<4x8xf32>) -> tensor<32xf32>
    %194 = stablehlo.dot_general %193, %arg24, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<32x32xf32>) -> tensor<32xf32>
    %195 = stablehlo.add %121, %194 : tensor<32xf32>
    %196 = stablehlo.reduce(%195 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %197 = stablehlo.divide %196, %7 : tensor<f32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %199 = stablehlo.subtract %195, %198 : tensor<32xf32>
    %200 = stablehlo.multiply %199, %199 : tensor<32xf32>
    %201 = stablehlo.reduce(%200 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %202 = stablehlo.divide %201, %7 : tensor<f32>
    %203 = stablehlo.add %202, %6 : tensor<f32>
    %204 = stablehlo.rsqrt %203 : tensor<f32>
    %205 = stablehlo.broadcast_in_dim %204, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %206 = stablehlo.multiply %199, %205 : tensor<32xf32>
    %207 = stablehlo.multiply %206, %arg21 : tensor<32xf32>
    %208 = stablehlo.add %207, %arg31 : tensor<32xf32>
    %209 = stablehlo.dot_general %208, %arg19, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<64x32xf32>) -> tensor<64xf32>
    %210 = stablehlo.dot_general %208, %arg22, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<64x32xf32>) -> tensor<64xf32>
    %211 = stablehlo.negate %209 : tensor<64xf32>
    %212 = stablehlo.exponential %211 : tensor<64xf32>
    %213 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %214 = stablehlo.add %213, %212 : tensor<64xf32>
    %215 = stablehlo.divide %213, %214 : tensor<64xf32>
    %216 = stablehlo.multiply %209, %215 : tensor<64xf32>
    %217 = stablehlo.multiply %216, %210 : tensor<64xf32>
    %218 = stablehlo.dot_general %217, %arg18, contracting_dims = [0] x [1] : (tensor<64xf32>, tensor<32x64xf32>) -> tensor<32xf32>
    %219 = stablehlo.add %195, %218 : tensor<32xf32>
    %220 = stablehlo.reduce(%219 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %221 = stablehlo.divide %220, %7 : tensor<f32>
    %222 = stablehlo.broadcast_in_dim %221, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %223 = stablehlo.subtract %219, %222 : tensor<32xf32>
    %224 = stablehlo.multiply %223, %223 : tensor<32xf32>
    %225 = stablehlo.reduce(%224 init: %2) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %226 = stablehlo.divide %225, %7 : tensor<f32>
    %227 = stablehlo.add %226, %6 : tensor<f32>
    %228 = stablehlo.rsqrt %227 : tensor<f32>
    %229 = stablehlo.broadcast_in_dim %228, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %230 = stablehlo.multiply %223, %229 : tensor<32xf32>
    %231 = stablehlo.multiply %230, %arg1 : tensor<32xf32>
    %232 = stablehlo.add %231, %arg2 : tensor<32xf32>
    %233 = stablehlo.dot_general %232, %arg3, contracting_dims = [0] x [1] : (tensor<32xf32>, tensor<32x32xf32>) -> tensor<32xf32>
    return %233, %169, %171 : tensor<32xf32>, tensor<2x256x2x8xf32>, tensor<2x256x2x8xf32>
  }
}
