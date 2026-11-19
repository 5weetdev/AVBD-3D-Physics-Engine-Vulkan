/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#pragma once

// Host mirrors of every "*ArgsGpu" struct dispatched to a Slang compute kernel through
// HostRingBuffer::write() + a device address, one header per shader family under args/. Each
// struct is a byte-for-byte transcription of its Slang counterpart, with VkDeviceAddress standing
// in for pointers and explicit _padN fields for the padding Slang inserts under scalar layout.
// A struct's size check alone cannot catch a shifted field: avbd_test_vk_layout compares every
// field offset against the Slang compiler's (engine/shaders/layout_probe.slang).

#include "args/broadphase_args.h"
#include "args/narrowphase_args.h"
#include "args/constraints_args.h"
#include "args/primitives_args.h"
#include "args/solver_args.h"
#include "args/render_args.h"
#include "args/interact_args.h"
#include "args/sleep_args.h"
#include "args/wake_args.h"
#include "args/mutate_args.h"
