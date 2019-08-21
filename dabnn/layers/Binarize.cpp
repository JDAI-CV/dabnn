// Copyright 2019 JD.com Inc. JD AI

#include "Binarize.h"

#include <dabnn/bitpack.h>
#include <dabnn/net.h>

namespace bnn {
void Binarize::forward_impl() const { pack_mat(*input_mat, *output_mat); }

}  // namespace bnn
