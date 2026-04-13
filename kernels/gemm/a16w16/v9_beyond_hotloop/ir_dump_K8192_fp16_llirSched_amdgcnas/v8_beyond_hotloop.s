	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v8_beyond_hotloop               ; -- Begin function v8_beyond_hotloop
	.p2align	8
	.type	v8_beyond_hotloop,@function
v8_beyond_hotloop:                      ; @v8_beyond_hotloop
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.3:
s_load_dwordx2 s[2:3], s[0:1], 0x0
s_load_dwordx8 s[4:11], s[0:1], 0x8
s_load_dwordx4 s[12:15], s[0:1], 0x28
s_waitcnt lgkmcnt(0)
s_branch .LBB0_0
.p2align 8
; %bb.4:
.LBB0_0:
v_readfirstlane_b32 s13, v0
s_bfe_u32 s21, s13, 0x20006
s_add_i32 s0, s8, 0xff
s_ashr_i32 s1, s0, 31
s_lshr_b32 s1, s1, 24
s_add_i32 s0, s0, s1
s_ashr_i32 s0, s0, 8
s_add_i32 s1, s9, 0xff
s_ashr_i32 s8, s1, 31
s_lshr_b32 s8, s8, 24
s_add_i32 s1, s1, s8
s_ashr_i32 s1, s1, 8
s_ashr_i32 s8, s16, 31
s_lshr_b32 s8, s8, 29
s_add_i32 s8, s16, s8
s_ashr_i32 s8, s8, 3
s_lshl_b32 s9, s16, 5
s_mulk_i32 s8, 0xff01
s_add_i32 s8, s8, s9
s_lshl_b32 s9, s1, 2
s_xor_b32 s1, s8, s1
s_ashr_i32 s1, s1, 31
s_abs_i32 s14, s8
s_abs_i32 s15, s9
v_cvt_f32_u32_e32 v1, s15
v_rcp_iflag_f32_e32 v1, v1
s_nop 0
v_mul_f32_e32 v1, 0x4f7ffffe, v1
v_cvt_u32_f32_e32 v1, v1
s_sub_i32 s16, 0, s15
v_readfirstlane_b32 s17, v1
s_mul_i32 s16, s16, s17
s_mul_hi_u32 s16, s17, s16
s_add_i32 s17, s17, s16
s_mul_hi_u32 s16, s14, s17
s_mul_i32 s17, s16, s15
s_sub_i32 s14, s14, s17
s_add_i32 s17, s16, 1
s_sub_i32 s18, s14, s15
s_cmp_ge_u32 s14, s15
s_cselect_b32 s16, s17, s16
s_cselect_b32 s14, s18, s14
s_add_i32 s17, s16, 1
s_cmp_ge_u32 s14, s15
s_cselect_b32 s14, s17, s16
s_xor_b32 s14, s14, s1
s_sub_i32 s1, s14, s1
s_lshl_b32 s14, s1, 2
s_sub_i32 s0, s0, s14
s_min_i32 s0, s0, 4
s_mul_i32 s1, s1, s9
s_sub_i32 s1, s8, s1
s_xor_b32 s8, s1, s0
s_ashr_i32 s8, s8, 31
s_abs_i32 s9, s1
s_abs_i32 s15, s0
v_cvt_f32_u32_e32 v1, s15
v_rcp_iflag_f32_e32 v1, v1
s_nop 0
v_mul_f32_e32 v1, 0x4f7ffffe, v1
v_cvt_u32_f32_e32 v1, v1
s_sub_i32 s16, 0, s15
v_readfirstlane_b32 s17, v1
s_mul_i32 s16, s16, s17
s_mul_hi_u32 s16, s17, s16
s_add_i32 s17, s17, s16
s_mul_hi_u32 s16, s9, s17
s_mul_i32 s17, s16, s15
s_sub_i32 s9, s9, s17
s_add_i32 s17, s16, 1
s_sub_i32 s18, s9, s15
s_cmp_ge_u32 s9, s15
s_cselect_b32 s16, s17, s16
s_cselect_b32 s9, s18, s9
s_add_i32 s17, s16, 1
s_cmp_ge_u32 s9, s15
s_cselect_b32 s9, s17, s16
s_xor_b32 s9, s9, s8
s_sub_i32 s8, s9, s8
s_mul_i32 s0, s8, s0
s_sub_i32 s0, s1, s0
s_add_i32 s0, s0, s14
v_and_b32_e32 v1, 63, v0
v_lshl_or_b32 v1, s21, 6, v1
v_lshlrev_b32_e32 v2, 1, v0
v_and_b32_e32 v2, 0x70, v2
v_or_b32_e32 v3, s21, v2
v_or_b32_e32 v4, 4, v3
v_or_b32_e32 v5, 8, v3
v_or_b32_e32 v6, 12, v3
v_or_b32_e32 v7, 0x80, v3
v_or_b32_e32 v8, 0x84, v3
v_or_b32_e32 v9, 0x88, v3
v_or_b32_e32 v10, 0x8c, v3
v_lshlrev_b32_e32 v2, 3, v0
v_and_b32_e32 v12, 56, v2
s_lshl_b32 s9, s0, 8
s_mul_i32 s14, s9, s10
s_ashr_i32 s15, s14, 31
s_lshl_b64 s[14:15], s[14:15], 1
s_add_u32 s16, s2, s14
s_addc_u32 s24, s3, s15
s_lshl_b32 s8, s8, 8
s_mul_i32 s14, s8, s11
s_ashr_i32 s15, s14, 31
s_lshl_b64 s[14:15], s[14:15], 1
s_add_u32 s20, s4, s14
s_addc_u32 s1, s5, s15
v_mul_lo_u32 v11, v3, s10
v_mul_lo_u32 v13, v4, s10
v_mul_lo_u32 v14, v5, s10
v_mul_lo_u32 v15, v6, s10
v_mul_lo_u32 v24, v7, s10
v_mul_lo_u32 v25, v8, s10
v_mul_lo_u32 v26, v9, s10
v_mul_lo_u32 v27, v10, s10
v_mad_u64_u32 v[16:17], s[4:5], v3, s11, v[12:13]
v_mad_u64_u32 v[18:19], s[4:5], v4, s11, v[12:13]
v_mad_u64_u32 v[20:21], s[4:5], v5, s11, v[12:13]
v_mad_u64_u32 v[22:23], s[4:5], v6, s11, v[12:13]
s_lshl_b32 s4, s11, 8
s_ashr_i32 s5, s4, 1
s_and_b32 s17, s24, 0xffff
s_mov_b32 s19, 0x27000
s_mov_b32 s18, 0x7ffffffe
s_mul_i32 s50, s21, 0x420
s_add_i32 s4, s50, 0
v_add_lshl_u32 v4, v11, v12, 1
s_mov_b32 m0, s4
s_nop 0
buffer_load_dwordx4 v4, s[16:19], 0, offen, lds
s_add_i32 s51, s50, 0x1080
s_add_i32 s11, s4, 0x1080
v_add_lshl_u32 v5, v13, v12, 1
s_mov_b32 m0, s11
s_nop 0
buffer_load_dwordx4 v5, s[16:19], 0, offen, lds
s_add_i32 s52, s50, 0x2100
s_add_i32 s14, s4, 0x2100
v_add_lshl_u32 v6, v14, v12, 1
s_mov_b32 m0, s14
s_nop 0
buffer_load_dwordx4 v6, s[16:19], 0, offen, lds
s_add_i32 s53, s50, 0x3180
s_add_i32 s15, s4, 0x3180
v_add_lshl_u32 v7, v15, v12, 1
s_mov_b32 m0, s15
s_nop 0
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
s_add_i32 s28, s4, 0x4200
v_add_lshl_u32 v8, v24, v12, 1
s_mov_b32 m0, s28
s_nop 0
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
s_add_i32 s29, s4, 0x5280
v_add_lshl_u32 v9, v25, v12, 1
s_mov_b32 m0, s29
s_nop 0
buffer_load_dwordx4 v9, s[16:19], 0, offen, lds
s_add_i32 s30, s4, 0x6300
v_add_lshl_u32 v10, v26, v12, 1
s_mov_b32 m0, s30
s_nop 0
buffer_load_dwordx4 v10, s[16:19], 0, offen, lds
s_add_i32 s31, s4, 0x7380
v_add_lshl_u32 v11, v27, v12, 1
s_mov_b32 m0, s31
s_nop 0
buffer_load_dwordx4 v11, s[16:19], 0, offen, lds
s_and_b32 s21, s1, 0xffff
s_mov_b32 s22, s18
s_mov_b32 s23, s19
s_add_i32 s54, 0, 0x107e0
s_add_i32 s33, s54, s50
v_lshlrev_b32_e32 v12, 1, v16
s_mov_b32 m0, s33
s_nop 0
buffer_load_dwordx4 v12, s[20:23], 0, offen, lds
s_add_i32 s34, s54, s51
v_lshlrev_b32_e32 v13, 1, v18
s_mov_b32 m0, s34
s_nop 0
buffer_load_dwordx4 v13, s[20:23], 0, offen, lds
s_add_i32 s35, s54, s52
v_lshlrev_b32_e32 v14, 1, v20
s_mov_b32 m0, s35
s_nop 0
buffer_load_dwordx4 v14, s[20:23], 0, offen, lds
s_add_i32 s36, s54, s53
v_lshlrev_b32_e32 v15, 1, v22
s_mov_b32 m0, s36
s_nop 0
buffer_load_dwordx4 v15, s[20:23], 0, offen, lds
s_add_i32 s40, 0, 0x18bc0
s_add_i32 s37, s40, s50
v_add_lshl_u32 v16, v16, s5, 1
s_mov_b32 m0, s37
s_nop 0
buffer_load_dwordx4 v16, s[20:23], 0, offen, lds
s_add_i32 s38, s40, s51
v_add_lshl_u32 v17, v18, s5, 1
s_mov_b32 m0, s38
s_nop 0
buffer_load_dwordx4 v17, s[20:23], 0, offen, lds
s_add_i32 s39, s40, s52
v_add_lshl_u32 v18, v20, s5, 1
s_mov_b32 m0, s39
s_nop 0
buffer_load_dwordx4 v18, s[20:23], 0, offen, lds
s_add_i32 s40, s40, s53
v_add_lshl_u32 v20, v22, s5, 1
s_mov_b32 m0, s40
s_nop 0
buffer_load_dwordx4 v20, s[20:23], 0, offen, lds
s_add_u32 s16, s16, 0x80
s_addc_u32 s5, s24, 0
s_add_u32 s24, s20, 0x80
s_addc_u32 s25, s1, 0
s_waitcnt lgkmcnt(0)
s_barrier
s_and_b32 s17, s5, 0xffff
s_add_i32 s21, s37, 0xfffef840
s_mov_b32 m0, s21
s_nop 0
buffer_load_dwordx4 v4, s[16:19], 0, offen, lds
s_add_i32 s22, s37, 0xffff08c0
s_mov_b32 m0, s22
s_nop 0
buffer_load_dwordx4 v5, s[16:19], 0, offen, lds
s_add_i32 s23, s37, 0xffff1940
s_mov_b32 m0, s23
s_nop 0
buffer_load_dwordx4 v6, s[16:19], 0, offen, lds
s_add_i32 s41, s37, 0xffff29c0
s_mov_b32 m0, s41
s_nop 0
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
s_add_i32 s42, s37, 0xffff3a40
s_mov_b32 m0, s42
s_nop 0
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
s_add_i32 s43, s37, 0xffff4ac0
s_mov_b32 m0, s43
s_nop 0
buffer_load_dwordx4 v9, s[16:19], 0, offen, lds
s_add_i32 s44, s37, 0xffff5b40
s_mov_b32 m0, s44
s_nop 0
buffer_load_dwordx4 v10, s[16:19], 0, offen, lds
s_add_i32 s45, s37, 0xffff6bc0
s_mov_b32 m0, s45
s_nop 0
buffer_load_dwordx4 v11, s[16:19], 0, offen, lds
s_and_b32 s25, s25, 0xffff
s_mov_b32 s26, s18
s_mov_b32 s27, s19
s_add_i32 s49, 0, 0x149e0
s_add_i32 s46, s49, s50
s_mov_b32 m0, s46
s_nop 0
buffer_load_dwordx4 v12, s[24:27], 0, offen, lds
s_add_i32 s47, s49, s51
s_mov_b32 m0, s47
s_nop 0
buffer_load_dwordx4 v13, s[24:27], 0, offen, lds
s_add_i32 s48, s49, s52
s_mov_b32 m0, s48
s_nop 0
buffer_load_dwordx4 v14, s[24:27], 0, offen, lds
s_add_i32 s49, s49, s53
s_mov_b32 m0, s49
s_nop 0
buffer_load_dwordx4 v15, s[24:27], 0, offen, lds
s_add_i32 s5, 0, 0x1cdc0
s_add_i32 s50, s5, s50
s_mov_b32 m0, s50
s_nop 0
buffer_load_dwordx4 v16, s[24:27], 0, offen, lds
s_add_i32 s51, s5, s51
s_mov_b32 m0, s51
s_nop 0
buffer_load_dwordx4 v17, s[24:27], 0, offen, lds
s_add_i32 s52, s5, s52
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v18, s[24:27], 0, offen, lds
s_add_i32 s53, s5, s53
s_mov_b32 m0, s53
s_nop 0
buffer_load_dwordx4 v20, s[24:27], 0, offen, lds
s_waitcnt vmcnt(20), lgkmcnt(0)
s_barrier
v_and_b32_e32 v21, 15, v0
v_lshlrev_b32_e32 v19, 10, v21
s_movk_i32 s5, 0xb0
v_and_or_b32 v3, v1, s5, v19
v_lshlrev_b32_e32 v22, 5, v21
v_add_u32_e32 v3, v3, v22
v_add_u32_e32 v3, 0, v3
ds_read_b128 v[82:85], v3
ds_read_b128 v[86:89], v3, offset:64
ds_read_b128 v[74:77], v3, offset:256
ds_read_b128 v[78:81], v3, offset:320
ds_read_b128 v[66:69], v3, offset:512
ds_read_b128 v[70:73], v3, offset:576
ds_read_b128 v[58:61], v3, offset:768
ds_read_b128 v[62:65], v3, offset:832
ds_read_b128 v[50:53], v3, offset:16896
ds_read_b128 v[54:57], v3, offset:16960
ds_read_b128 v[42:45], v3, offset:17152
ds_read_b128 v[46:49], v3, offset:17216
ds_read_b128 v[34:37], v3, offset:17408
ds_read_b128 v[38:41], v3, offset:17472
ds_read_b128 v[26:29], v3, offset:17664
ds_read_b128 v[30:33], v3, offset:17728
s_and_b32 s5, s13, 64
v_and_or_b32 v19, v0, 48, v19
v_add_u32_e32 v19, v19, v22
v_lshl_add_u32 v19, s5, 1, v19
v_add_u32_e32 v22, s54, v19
ds_read_b128 v[90:93], v22
ds_read_b128 v[98:101], v22, offset:64
ds_read_b128 v[94:97], v22, offset:256
ds_read_b128 v[102:105], v22, offset:320
ds_read_b128 v[106:109], v22, offset:512
ds_read_b128 v[110:113], v22, offset:576
ds_read_b128 v[114:117], v22, offset:768
ds_read_b128 v[118:121], v22, offset:832
s_add_u32 s20, s20, 0x180
s_addc_u32 s24, s1, 0
s_mul_i32 s10, s10, s0
s_lshl_b32 s0, s10, 8
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s0, s2, s0
s_addc_u32 s1, s3, s1
s_add_u32 s10, s0, 0x180
s_addc_u32 s25, s1, 0
v_mov_b32_e32 v122, 0
s_mov_b32 s26, -2
v_add_u32_e32 v19, 0, v19
v_add_u32_e32 v22, 0x18bc0, v19
v_add_u32_e32 v23, 0x149e0, v19
v_add_u32_e32 v24, 0x1cdc0, v19
v_add_u32_e32 v25, 0x107e0, v19
v_accvgpr_write_b32 a124, v122
v_accvgpr_write_b32 a125, v122
v_accvgpr_write_b32 a126, v122
v_accvgpr_write_b32 a127, v122
v_accvgpr_write_b32 a0, v122
v_accvgpr_write_b32 a1, v122
v_accvgpr_write_b32 a2, v122
v_accvgpr_write_b32 a3, v122
v_accvgpr_write_b32 a4, v122
v_accvgpr_write_b32 a5, v122
v_accvgpr_write_b32 a6, v122
v_accvgpr_write_b32 a7, v122
v_accvgpr_write_b32 a8, v122
v_accvgpr_write_b32 a9, v122
v_accvgpr_write_b32 a10, v122
v_accvgpr_write_b32 a11, v122
v_accvgpr_write_b32 a12, v122
v_accvgpr_write_b32 a13, v122
v_accvgpr_write_b32 a14, v122
v_accvgpr_write_b32 a15, v122
v_accvgpr_write_b32 a16, v122
v_accvgpr_write_b32 a17, v122
v_accvgpr_write_b32 a18, v122
v_accvgpr_write_b32 a19, v122
v_accvgpr_write_b32 a20, v122
v_accvgpr_write_b32 a21, v122
v_accvgpr_write_b32 a22, v122
v_accvgpr_write_b32 a23, v122
v_accvgpr_write_b32 a24, v122
v_accvgpr_write_b32 a25, v122
v_accvgpr_write_b32 a26, v122
v_accvgpr_write_b32 a27, v122
v_accvgpr_write_b32 a28, v122
v_accvgpr_write_b32 a29, v122
v_accvgpr_write_b32 a30, v122
v_accvgpr_write_b32 a31, v122
v_accvgpr_write_b32 a32, v122
v_accvgpr_write_b32 a33, v122
v_accvgpr_write_b32 a34, v122
v_accvgpr_write_b32 a35, v122
v_accvgpr_write_b32 a36, v122
v_accvgpr_write_b32 a37, v122
v_accvgpr_write_b32 a38, v122
v_accvgpr_write_b32 a39, v122
v_accvgpr_write_b32 a40, v122
v_accvgpr_write_b32 a41, v122
v_accvgpr_write_b32 a42, v122
v_accvgpr_write_b32 a43, v122
v_accvgpr_write_b32 a44, v122
v_accvgpr_write_b32 a45, v122
v_accvgpr_write_b32 a46, v122
v_accvgpr_write_b32 a47, v122
v_accvgpr_write_b32 a48, v122
v_accvgpr_write_b32 a49, v122
v_accvgpr_write_b32 a50, v122
v_accvgpr_write_b32 a51, v122
v_accvgpr_write_b32 a52, v122
v_accvgpr_write_b32 a53, v122
v_accvgpr_write_b32 a54, v122
v_accvgpr_write_b32 a55, v122
v_accvgpr_write_b32 a56, v122
v_accvgpr_write_b32 a57, v122
v_accvgpr_write_b32 a58, v122
v_accvgpr_write_b32 a59, v122
v_accvgpr_write_b32 a60, v122
v_accvgpr_write_b32 a61, v122
v_accvgpr_write_b32 a62, v122
v_accvgpr_write_b32 a63, v122
v_accvgpr_write_b32 a64, v122
v_accvgpr_write_b32 a65, v122
v_accvgpr_write_b32 a66, v122
v_accvgpr_write_b32 a67, v122
v_accvgpr_write_b32 a68, v122
v_accvgpr_write_b32 a69, v122
v_accvgpr_write_b32 a70, v122
v_accvgpr_write_b32 a71, v122
v_accvgpr_write_b32 a72, v122
v_accvgpr_write_b32 a73, v122
v_accvgpr_write_b32 a74, v122
v_accvgpr_write_b32 a75, v122
v_accvgpr_write_b32 a76, v122
v_accvgpr_write_b32 a77, v122
v_accvgpr_write_b32 a78, v122
v_accvgpr_write_b32 a79, v122
v_accvgpr_write_b32 a80, v122
v_accvgpr_write_b32 a81, v122
v_accvgpr_write_b32 a82, v122
v_accvgpr_write_b32 a83, v122
v_accvgpr_write_b32 a84, v122
v_accvgpr_write_b32 a85, v122
v_accvgpr_write_b32 a86, v122
v_accvgpr_write_b32 a87, v122
v_accvgpr_write_b32 a88, v122
v_accvgpr_write_b32 a89, v122
v_accvgpr_write_b32 a90, v122
v_accvgpr_write_b32 a91, v122
v_accvgpr_write_b32 a92, v122
v_accvgpr_write_b32 a93, v122
v_accvgpr_write_b32 a94, v122
v_accvgpr_write_b32 a95, v122
v_accvgpr_write_b32 a96, v122
v_accvgpr_write_b32 a97, v122
v_accvgpr_write_b32 a98, v122
v_accvgpr_write_b32 a99, v122
v_accvgpr_write_b32 a100, v122
v_accvgpr_write_b32 a101, v122
v_accvgpr_write_b32 a102, v122
v_accvgpr_write_b32 a103, v122
v_accvgpr_write_b32 a104, v122
v_accvgpr_write_b32 a105, v122
v_accvgpr_write_b32 a106, v122
v_accvgpr_write_b32 a107, v122
v_accvgpr_write_b32 a108, v122
v_accvgpr_write_b32 a109, v122
v_accvgpr_write_b32 a110, v122
v_accvgpr_write_b32 a111, v122
v_accvgpr_write_b32 a112, v122
v_accvgpr_write_b32 a113, v122
v_accvgpr_write_b32 a114, v122
v_accvgpr_write_b32 a115, v122
v_accvgpr_write_b32 a116, v122
v_accvgpr_write_b32 a117, v122
v_accvgpr_write_b32 a118, v122
v_accvgpr_write_b32 a119, v122
v_accvgpr_write_b32 a120, v122
v_accvgpr_write_b32 a121, v122
v_accvgpr_write_b32 a122, v122
v_accvgpr_write_b32 a123, v122
v_accvgpr_write_b32 a252, v122
v_accvgpr_write_b32 a253, v122
v_accvgpr_write_b32 a254, v122
v_accvgpr_write_b32 a255, v122
v_accvgpr_write_b32 a128, v122
v_accvgpr_write_b32 a129, v122
v_accvgpr_write_b32 a130, v122
v_accvgpr_write_b32 a131, v122
v_accvgpr_write_b32 a132, v122
v_accvgpr_write_b32 a133, v122
v_accvgpr_write_b32 a134, v122
v_accvgpr_write_b32 a135, v122
v_accvgpr_write_b32 a136, v122
v_accvgpr_write_b32 a137, v122
v_accvgpr_write_b32 a138, v122
v_accvgpr_write_b32 a139, v122
v_accvgpr_write_b32 a140, v122
v_accvgpr_write_b32 a141, v122
v_accvgpr_write_b32 a142, v122
v_accvgpr_write_b32 a143, v122
v_accvgpr_write_b32 a144, v122
v_accvgpr_write_b32 a145, v122
v_accvgpr_write_b32 a146, v122
v_accvgpr_write_b32 a147, v122
v_accvgpr_write_b32 a148, v122
v_accvgpr_write_b32 a149, v122
v_accvgpr_write_b32 a150, v122
v_accvgpr_write_b32 a151, v122
v_accvgpr_write_b32 a152, v122
v_accvgpr_write_b32 a153, v122
v_accvgpr_write_b32 a154, v122
v_accvgpr_write_b32 a155, v122
v_accvgpr_write_b32 a156, v122
v_accvgpr_write_b32 a157, v122
v_accvgpr_write_b32 a158, v122
v_accvgpr_write_b32 a159, v122
v_accvgpr_write_b32 a160, v122
v_accvgpr_write_b32 a161, v122
v_accvgpr_write_b32 a162, v122
v_accvgpr_write_b32 a163, v122
v_accvgpr_write_b32 a164, v122
v_accvgpr_write_b32 a165, v122
v_accvgpr_write_b32 a166, v122
v_accvgpr_write_b32 a167, v122
v_accvgpr_write_b32 a168, v122
v_accvgpr_write_b32 a169, v122
v_accvgpr_write_b32 a170, v122
v_accvgpr_write_b32 a171, v122
v_accvgpr_write_b32 a172, v122
v_accvgpr_write_b32 a173, v122
v_accvgpr_write_b32 a174, v122
v_accvgpr_write_b32 a175, v122
v_accvgpr_write_b32 a176, v122
v_accvgpr_write_b32 a177, v122
v_accvgpr_write_b32 a178, v122
v_accvgpr_write_b32 a179, v122
v_accvgpr_write_b32 a180, v122
v_accvgpr_write_b32 a181, v122
v_accvgpr_write_b32 a182, v122
v_accvgpr_write_b32 a183, v122
v_accvgpr_write_b32 a184, v122
v_accvgpr_write_b32 a185, v122
v_accvgpr_write_b32 a186, v122
v_accvgpr_write_b32 a187, v122
v_accvgpr_write_b32 a188, v122
v_accvgpr_write_b32 a189, v122
v_accvgpr_write_b32 a190, v122
v_accvgpr_write_b32 a191, v122
v_accvgpr_write_b32 a192, v122
v_accvgpr_write_b32 a193, v122
v_accvgpr_write_b32 a194, v122
v_accvgpr_write_b32 a195, v122
v_accvgpr_write_b32 a196, v122
v_accvgpr_write_b32 a197, v122
v_accvgpr_write_b32 a198, v122
v_accvgpr_write_b32 a199, v122
v_accvgpr_write_b32 a200, v122
v_accvgpr_write_b32 a201, v122
v_accvgpr_write_b32 a202, v122
v_accvgpr_write_b32 a203, v122
v_accvgpr_write_b32 a204, v122
v_accvgpr_write_b32 a205, v122
v_accvgpr_write_b32 a206, v122
v_accvgpr_write_b32 a207, v122
v_accvgpr_write_b32 a208, v122
v_accvgpr_write_b32 a209, v122
v_accvgpr_write_b32 a210, v122
v_accvgpr_write_b32 a211, v122
v_accvgpr_write_b32 a212, v122
v_accvgpr_write_b32 a213, v122
v_accvgpr_write_b32 a214, v122
v_accvgpr_write_b32 a215, v122
v_accvgpr_write_b32 a216, v122
v_accvgpr_write_b32 a217, v122
v_accvgpr_write_b32 a218, v122
v_accvgpr_write_b32 a219, v122
v_accvgpr_write_b32 a220, v122
v_accvgpr_write_b32 a221, v122
v_accvgpr_write_b32 a222, v122
v_accvgpr_write_b32 a223, v122
v_accvgpr_write_b32 a224, v122
v_accvgpr_write_b32 a225, v122
v_accvgpr_write_b32 a226, v122
v_accvgpr_write_b32 a227, v122
v_accvgpr_write_b32 a228, v122
v_accvgpr_write_b32 a229, v122
v_accvgpr_write_b32 a230, v122
v_accvgpr_write_b32 a231, v122
v_accvgpr_write_b32 a232, v122
v_accvgpr_write_b32 a233, v122
v_accvgpr_write_b32 a234, v122
v_accvgpr_write_b32 a235, v122
v_accvgpr_write_b32 a236, v122
v_accvgpr_write_b32 a237, v122
v_accvgpr_write_b32 a238, v122
v_accvgpr_write_b32 a239, v122
v_accvgpr_write_b32 a240, v122
v_accvgpr_write_b32 a241, v122
v_accvgpr_write_b32 a242, v122
v_accvgpr_write_b32 a243, v122
v_accvgpr_write_b32 a244, v122
v_accvgpr_write_b32 a245, v122
v_accvgpr_write_b32 a246, v122
v_accvgpr_write_b32 a247, v122
v_accvgpr_write_b32 a248, v122
v_accvgpr_write_b32 a249, v122
v_accvgpr_write_b32 a250, v122
v_accvgpr_write_b32 a251, v122
s_mov_b32 s2, s18
s_mov_b32 s3, s19
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
.LBB0_1:
v_mfma_f32_16x16x32_f16 a[252:255], v[90:93], v[82:85], a[252:255]
s_add_u32 s0, s20, 0xffffff80
s_addc_u32 s1, s24, -1
s_add_u32 s16, s10, 0xffffff80
s_addc_u32 s27, s25, -1
v_mfma_f32_16x16x32_f16 a[252:255], v[98:101], v[86:89], a[252:255]
v_mfma_f32_16x16x32_f16 a[128:131], v[94:97], v[82:85], a[128:131]
v_mfma_f32_16x16x32_f16 a[128:131], v[102:105], v[86:89], a[128:131]
ds_read_b128 v[122:125], v22
v_mfma_f32_16x16x32_f16 a[132:135], v[106:109], v[82:85], a[132:135]
ds_read_b128 v[126:129], v22, offset:64
v_mfma_f32_16x16x32_f16 a[132:135], v[110:113], v[86:89], a[132:135]
ds_read_b128 v[130:133], v22, offset:256
v_mfma_f32_16x16x32_f16 a[136:139], v[114:117], v[82:85], a[136:139]
ds_read_b128 v[134:137], v22, offset:320
v_mfma_f32_16x16x32_f16 a[136:139], v[118:121], v[86:89], a[136:139]
ds_read_b128 v[138:141], v22, offset:512
v_mfma_f32_16x16x32_f16 a[140:143], v[90:93], v[74:77], a[140:143]
ds_read_b128 v[142:145], v22, offset:576
v_mfma_f32_16x16x32_f16 a[140:143], v[98:101], v[78:81], a[140:143]
ds_read_b128 v[146:149], v22, offset:768
v_mfma_f32_16x16x32_f16 a[144:147], v[94:97], v[74:77], a[144:147]
ds_read_b128 v[150:153], v22, offset:832
v_mfma_f32_16x16x32_f16 a[144:147], v[102:105], v[78:81], a[144:147]
v_mfma_f32_16x16x32_f16 a[148:151], v[106:109], v[74:77], a[148:151]
v_mfma_f32_16x16x32_f16 a[148:151], v[110:113], v[78:81], a[148:151]
s_and_b32 s17, s27, 0xffff
s_mov_b32 m0, s4
v_mfma_f32_16x16x32_f16 a[152:155], v[114:117], v[74:77], a[152:155]
buffer_load_dwordx4 v4, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[152:155], v[118:121], v[78:81], a[152:155]
v_mfma_f32_16x16x32_f16 a[156:159], v[90:93], v[66:69], a[156:159]
v_mfma_f32_16x16x32_f16 a[156:159], v[98:101], v[70:73], a[156:159]
s_mov_b32 m0, s11
v_mfma_f32_16x16x32_f16 a[160:163], v[94:97], v[66:69], a[160:163]
buffer_load_dwordx4 v5, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[160:163], v[102:105], v[70:73], a[160:163]
v_mfma_f32_16x16x32_f16 a[164:167], v[106:109], v[66:69], a[164:167]
v_mfma_f32_16x16x32_f16 a[164:167], v[110:113], v[70:73], a[164:167]
s_mov_b32 m0, s14
v_mfma_f32_16x16x32_f16 a[168:171], v[114:117], v[66:69], a[168:171]
buffer_load_dwordx4 v6, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[168:171], v[118:121], v[70:73], a[168:171]
v_mfma_f32_16x16x32_f16 a[172:175], v[90:93], v[58:61], a[172:175]
v_mfma_f32_16x16x32_f16 a[172:175], v[98:101], v[62:65], a[172:175]
s_mov_b32 m0, s15
v_mfma_f32_16x16x32_f16 a[176:179], v[94:97], v[58:61], a[176:179]
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[176:179], v[102:105], v[62:65], a[176:179]
v_mfma_f32_16x16x32_f16 a[180:183], v[106:109], v[58:61], a[180:183]
v_mfma_f32_16x16x32_f16 a[180:183], v[110:113], v[62:65], a[180:183]
s_mov_b32 m0, s28
v_mfma_f32_16x16x32_f16 a[184:187], v[114:117], v[58:61], a[184:187]
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[184:187], v[118:121], v[62:65], a[184:187]
v_mfma_f32_16x16x32_f16 a[188:191], v[90:93], v[50:53], a[188:191]
v_mfma_f32_16x16x32_f16 a[188:191], v[98:101], v[54:57], a[188:191]
s_mov_b32 m0, s29
v_mfma_f32_16x16x32_f16 a[192:195], v[94:97], v[50:53], a[192:195]
buffer_load_dwordx4 v9, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[192:195], v[102:105], v[54:57], a[192:195]
v_mfma_f32_16x16x32_f16 a[196:199], v[106:109], v[50:53], a[196:199]
v_mfma_f32_16x16x32_f16 a[196:199], v[110:113], v[54:57], a[196:199]
s_mov_b32 m0, s30
v_mfma_f32_16x16x32_f16 a[200:203], v[114:117], v[50:53], a[200:203]
buffer_load_dwordx4 v10, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[200:203], v[118:121], v[54:57], a[200:203]
v_mfma_f32_16x16x32_f16 a[204:207], v[90:93], v[42:45], a[204:207]
v_mfma_f32_16x16x32_f16 a[204:207], v[98:101], v[46:49], a[204:207]
s_mov_b32 m0, s31
v_mfma_f32_16x16x32_f16 a[208:211], v[94:97], v[42:45], a[208:211]
buffer_load_dwordx4 v11, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[208:211], v[102:105], v[46:49], a[208:211]
v_mfma_f32_16x16x32_f16 a[212:215], v[106:109], v[42:45], a[212:215]
v_mfma_f32_16x16x32_f16 a[212:215], v[110:113], v[46:49], a[212:215]
s_and_b32 s1, s1, 0xffff
s_mov_b32 m0, s33
v_mfma_f32_16x16x32_f16 a[216:219], v[114:117], v[42:45], a[216:219]
buffer_load_dwordx4 v12, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[216:219], v[118:121], v[46:49], a[216:219]
v_mfma_f32_16x16x32_f16 a[220:223], v[90:93], v[34:37], a[220:223]
v_mfma_f32_16x16x32_f16 a[220:223], v[98:101], v[38:41], a[220:223]
s_mov_b32 m0, s34
v_mfma_f32_16x16x32_f16 a[224:227], v[94:97], v[34:37], a[224:227]
buffer_load_dwordx4 v13, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[224:227], v[102:105], v[38:41], a[224:227]
v_mfma_f32_16x16x32_f16 a[228:231], v[106:109], v[34:37], a[228:231]
v_mfma_f32_16x16x32_f16 a[228:231], v[110:113], v[38:41], a[228:231]
s_mov_b32 m0, s35
v_mfma_f32_16x16x32_f16 a[232:235], v[114:117], v[34:37], a[232:235]
buffer_load_dwordx4 v14, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[232:235], v[118:121], v[38:41], a[232:235]
v_mfma_f32_16x16x32_f16 a[236:239], v[90:93], v[26:29], a[236:239]
v_mfma_f32_16x16x32_f16 a[236:239], v[98:101], v[30:33], a[236:239]
s_mov_b32 m0, s36
v_mfma_f32_16x16x32_f16 a[240:243], v[94:97], v[26:29], a[240:243]
buffer_load_dwordx4 v15, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[240:243], v[102:105], v[30:33], a[240:243]
v_mfma_f32_16x16x32_f16 a[244:247], v[106:109], v[26:29], a[244:247]
v_mfma_f32_16x16x32_f16 a[244:247], v[110:113], v[30:33], a[244:247]
v_mfma_f32_16x16x32_f16 a[248:251], v[114:117], v[26:29], a[248:251]
s_waitcnt vmcnt(16), lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[248:251], v[118:121], v[30:33], a[248:251]
s_barrier
v_mfma_f32_16x16x32_f16 a[124:127], v[122:125], v[82:85], a[124:127]
v_mfma_f32_16x16x32_f16 a[124:127], v[126:129], v[86:89], a[124:127]
v_mfma_f32_16x16x32_f16 a[0:3], v[130:133], v[82:85], a[0:3]
v_mfma_f32_16x16x32_f16 a[0:3], v[134:137], v[86:89], a[0:3]
v_mfma_f32_16x16x32_f16 a[4:7], v[138:141], v[82:85], a[4:7]
v_mfma_f32_16x16x32_f16 a[4:7], v[142:145], v[86:89], a[4:7]
v_mfma_f32_16x16x32_f16 a[8:11], v[146:149], v[82:85], a[8:11]
v_mfma_f32_16x16x32_f16 a[8:11], v[150:153], v[86:89], a[8:11]
v_mfma_f32_16x16x32_f16 a[12:15], v[122:125], v[74:77], a[12:15]
v_mfma_f32_16x16x32_f16 a[12:15], v[126:129], v[78:81], a[12:15]
v_mfma_f32_16x16x32_f16 a[16:19], v[130:133], v[74:77], a[16:19]
v_mfma_f32_16x16x32_f16 a[16:19], v[134:137], v[78:81], a[16:19]
v_mfma_f32_16x16x32_f16 a[20:23], v[138:141], v[74:77], a[20:23]
v_mfma_f32_16x16x32_f16 a[20:23], v[142:145], v[78:81], a[20:23]
v_mfma_f32_16x16x32_f16 a[24:27], v[146:149], v[74:77], a[24:27]
v_mfma_f32_16x16x32_f16 a[24:27], v[150:153], v[78:81], a[24:27]
v_mfma_f32_16x16x32_f16 a[28:31], v[122:125], v[66:69], a[28:31]
v_mfma_f32_16x16x32_f16 a[28:31], v[126:129], v[70:73], a[28:31]
v_mfma_f32_16x16x32_f16 a[32:35], v[130:133], v[66:69], a[32:35]
v_mfma_f32_16x16x32_f16 a[32:35], v[134:137], v[70:73], a[32:35]
ds_read_b128 v[74:77], v3, offset:33792
v_mfma_f32_16x16x32_f16 a[36:39], v[138:141], v[66:69], a[36:39]
ds_read_b128 v[78:81], v3, offset:33856
v_mfma_f32_16x16x32_f16 a[36:39], v[142:145], v[70:73], a[36:39]
ds_read_b128 v[82:85], v3, offset:34048
v_mfma_f32_16x16x32_f16 a[40:43], v[146:149], v[66:69], a[40:43]
ds_read_b128 v[66:69], v3, offset:34112
v_mfma_f32_16x16x32_f16 a[40:43], v[150:153], v[70:73], a[40:43]
ds_read_b128 v[70:73], v3, offset:34304
v_mfma_f32_16x16x32_f16 a[44:47], v[122:125], v[58:61], a[44:47]
ds_read_b128 v[90:93], v3, offset:34368
v_mfma_f32_16x16x32_f16 a[44:47], v[126:129], v[62:65], a[44:47]
ds_read_b128 v[94:97], v3, offset:34560
v_mfma_f32_16x16x32_f16 a[48:51], v[130:133], v[58:61], a[48:51]
ds_read_b128 v[98:101], v3, offset:34624
v_mfma_f32_16x16x32_f16 a[48:51], v[134:137], v[62:65], a[48:51]
ds_read_b128 v[102:105], v3, offset:50688
v_mfma_f32_16x16x32_f16 a[52:55], v[138:141], v[58:61], a[52:55]
ds_read_b128 v[106:109], v3, offset:50752
v_mfma_f32_16x16x32_f16 a[52:55], v[142:145], v[62:65], a[52:55]
ds_read_b128 v[154:157], v3, offset:50944
v_mfma_f32_16x16x32_f16 a[56:59], v[146:149], v[58:61], a[56:59]
ds_read_b128 v[158:161], v3, offset:51008
v_mfma_f32_16x16x32_f16 a[56:59], v[150:153], v[62:65], a[56:59]
ds_read_b128 v[162:165], v3, offset:51200
v_mfma_f32_16x16x32_f16 a[60:63], v[122:125], v[50:53], a[60:63]
ds_read_b128 v[166:169], v3, offset:51264
v_mfma_f32_16x16x32_f16 a[60:63], v[126:129], v[54:57], a[60:63]
ds_read_b128 v[170:173], v3, offset:51456
v_mfma_f32_16x16x32_f16 a[64:67], v[130:133], v[50:53], a[64:67]
ds_read_b128 v[174:177], v3, offset:51520
v_mfma_f32_16x16x32_f16 a[64:67], v[134:137], v[54:57], a[64:67]
ds_read_b128 v[58:61], v23
v_mfma_f32_16x16x32_f16 a[68:71], v[138:141], v[50:53], a[68:71]
ds_read_b128 v[62:65], v23, offset:64
v_mfma_f32_16x16x32_f16 a[68:71], v[142:145], v[54:57], a[68:71]
ds_read_b128 v[86:89], v23, offset:256
v_mfma_f32_16x16x32_f16 a[72:75], v[146:149], v[50:53], a[72:75]
ds_read_b128 v[50:53], v23, offset:320
v_mfma_f32_16x16x32_f16 a[72:75], v[150:153], v[54:57], a[72:75]
ds_read_b128 v[54:57], v23, offset:512
v_mfma_f32_16x16x32_f16 a[76:79], v[122:125], v[42:45], a[76:79]
ds_read_b128 v[110:113], v23, offset:576
v_mfma_f32_16x16x32_f16 a[76:79], v[126:129], v[46:49], a[76:79]
ds_read_b128 v[114:117], v23, offset:768
v_mfma_f32_16x16x32_f16 a[80:83], v[130:133], v[42:45], a[80:83]
ds_read_b128 v[118:121], v23, offset:832
v_mfma_f32_16x16x32_f16 a[80:83], v[134:137], v[46:49], a[80:83]
v_mfma_f32_16x16x32_f16 a[84:87], v[138:141], v[42:45], a[84:87]
v_mfma_f32_16x16x32_f16 a[84:87], v[142:145], v[46:49], a[84:87]
s_mov_b32 m0, s37
v_mfma_f32_16x16x32_f16 a[88:91], v[146:149], v[42:45], a[88:91]
buffer_load_dwordx4 v16, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[88:91], v[150:153], v[46:49], a[88:91]
v_mfma_f32_16x16x32_f16 a[92:95], v[122:125], v[34:37], a[92:95]
v_mfma_f32_16x16x32_f16 a[92:95], v[126:129], v[38:41], a[92:95]
s_mov_b32 m0, s38
v_mfma_f32_16x16x32_f16 a[96:99], v[130:133], v[34:37], a[96:99]
buffer_load_dwordx4 v17, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[96:99], v[134:137], v[38:41], a[96:99]
v_mfma_f32_16x16x32_f16 a[100:103], v[138:141], v[34:37], a[100:103]
v_mfma_f32_16x16x32_f16 a[100:103], v[142:145], v[38:41], a[100:103]
s_mov_b32 m0, s39
v_mfma_f32_16x16x32_f16 a[104:107], v[146:149], v[34:37], a[104:107]
buffer_load_dwordx4 v18, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[104:107], v[150:153], v[38:41], a[104:107]
v_mfma_f32_16x16x32_f16 a[108:111], v[122:125], v[26:29], a[108:111]
v_mfma_f32_16x16x32_f16 a[108:111], v[126:129], v[30:33], a[108:111]
s_mov_b32 m0, s40
v_mfma_f32_16x16x32_f16 a[112:115], v[130:133], v[26:29], a[112:115]
buffer_load_dwordx4 v20, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[112:115], v[134:137], v[30:33], a[112:115]
v_mfma_f32_16x16x32_f16 a[116:119], v[138:141], v[26:29], a[116:119]
v_mfma_f32_16x16x32_f16 a[116:119], v[142:145], v[30:33], a[116:119]
v_mfma_f32_16x16x32_f16 a[120:123], v[146:149], v[26:29], a[120:123]
s_waitcnt vmcnt(16), lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[120:123], v[150:153], v[30:33], a[120:123]
s_barrier
v_mfma_f32_16x16x32_f16 a[252:255], v[58:61], v[74:77], a[252:255]
v_mfma_f32_16x16x32_f16 a[252:255], v[62:65], v[78:81], a[252:255]
v_mfma_f32_16x16x32_f16 a[128:131], v[86:89], v[74:77], a[128:131]
v_mfma_f32_16x16x32_f16 a[128:131], v[50:53], v[78:81], a[128:131]
ds_read_b128 v[122:125], v24
v_mfma_f32_16x16x32_f16 a[132:135], v[54:57], v[74:77], a[132:135]
ds_read_b128 v[126:129], v24, offset:64
v_mfma_f32_16x16x32_f16 a[132:135], v[110:113], v[78:81], a[132:135]
ds_read_b128 v[130:133], v24, offset:256
v_mfma_f32_16x16x32_f16 a[136:139], v[114:117], v[74:77], a[136:139]
ds_read_b128 v[134:137], v24, offset:320
v_mfma_f32_16x16x32_f16 a[136:139], v[118:121], v[78:81], a[136:139]
ds_read_b128 v[138:141], v24, offset:512
v_mfma_f32_16x16x32_f16 a[140:143], v[58:61], v[82:85], a[140:143]
ds_read_b128 v[142:145], v24, offset:576
v_mfma_f32_16x16x32_f16 a[140:143], v[62:65], v[66:69], a[140:143]
ds_read_b128 v[146:149], v24, offset:768
v_mfma_f32_16x16x32_f16 a[144:147], v[86:89], v[82:85], a[144:147]
ds_read_b128 v[150:153], v24, offset:832
v_mfma_f32_16x16x32_f16 a[144:147], v[50:53], v[66:69], a[144:147]
v_mfma_f32_16x16x32_f16 a[148:151], v[54:57], v[82:85], a[148:151]
v_mfma_f32_16x16x32_f16 a[148:151], v[110:113], v[66:69], a[148:151]
s_and_b32 s17, s25, 0xffff
s_mov_b32 s16, s10
s_mov_b32 m0, s21
v_mfma_f32_16x16x32_f16 a[152:155], v[114:117], v[82:85], a[152:155]
buffer_load_dwordx4 v4, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[152:155], v[118:121], v[66:69], a[152:155]
v_mfma_f32_16x16x32_f16 a[156:159], v[58:61], v[70:73], a[156:159]
v_mfma_f32_16x16x32_f16 a[156:159], v[62:65], v[90:93], a[156:159]
s_mov_b32 m0, s22
v_mfma_f32_16x16x32_f16 a[160:163], v[86:89], v[70:73], a[160:163]
buffer_load_dwordx4 v5, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[160:163], v[50:53], v[90:93], a[160:163]
v_mfma_f32_16x16x32_f16 a[164:167], v[54:57], v[70:73], a[164:167]
v_mfma_f32_16x16x32_f16 a[164:167], v[110:113], v[90:93], a[164:167]
s_mov_b32 m0, s23
v_mfma_f32_16x16x32_f16 a[168:171], v[114:117], v[70:73], a[168:171]
buffer_load_dwordx4 v6, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[168:171], v[118:121], v[90:93], a[168:171]
v_mfma_f32_16x16x32_f16 a[172:175], v[58:61], v[94:97], a[172:175]
v_mfma_f32_16x16x32_f16 a[172:175], v[62:65], v[98:101], a[172:175]
s_mov_b32 m0, s41
v_mfma_f32_16x16x32_f16 a[176:179], v[86:89], v[94:97], a[176:179]
buffer_load_dwordx4 v7, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[176:179], v[50:53], v[98:101], a[176:179]
v_mfma_f32_16x16x32_f16 a[180:183], v[54:57], v[94:97], a[180:183]
v_mfma_f32_16x16x32_f16 a[180:183], v[110:113], v[98:101], a[180:183]
s_mov_b32 m0, s42
v_mfma_f32_16x16x32_f16 a[184:187], v[114:117], v[94:97], a[184:187]
buffer_load_dwordx4 v8, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[184:187], v[118:121], v[98:101], a[184:187]
v_mfma_f32_16x16x32_f16 a[188:191], v[58:61], v[102:105], a[188:191]
v_mfma_f32_16x16x32_f16 a[188:191], v[62:65], v[106:109], a[188:191]
s_mov_b32 m0, s43
v_mfma_f32_16x16x32_f16 a[192:195], v[86:89], v[102:105], a[192:195]
buffer_load_dwordx4 v9, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[192:195], v[50:53], v[106:109], a[192:195]
v_mfma_f32_16x16x32_f16 a[196:199], v[54:57], v[102:105], a[196:199]
v_mfma_f32_16x16x32_f16 a[196:199], v[110:113], v[106:109], a[196:199]
s_mov_b32 m0, s44
v_mfma_f32_16x16x32_f16 a[200:203], v[114:117], v[102:105], a[200:203]
buffer_load_dwordx4 v10, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[200:203], v[118:121], v[106:109], a[200:203]
v_mfma_f32_16x16x32_f16 a[204:207], v[58:61], v[154:157], a[204:207]
v_mfma_f32_16x16x32_f16 a[204:207], v[62:65], v[158:161], a[204:207]
s_mov_b32 m0, s45
v_mfma_f32_16x16x32_f16 a[208:211], v[86:89], v[154:157], a[208:211]
buffer_load_dwordx4 v11, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[208:211], v[50:53], v[158:161], a[208:211]
v_mfma_f32_16x16x32_f16 a[212:215], v[54:57], v[154:157], a[212:215]
v_mfma_f32_16x16x32_f16 a[212:215], v[110:113], v[158:161], a[212:215]
s_and_b32 s17, s24, 0xffff
s_mov_b32 s16, s20
s_mov_b32 m0, s46
v_mfma_f32_16x16x32_f16 a[216:219], v[114:117], v[154:157], a[216:219]
buffer_load_dwordx4 v12, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[216:219], v[118:121], v[158:161], a[216:219]
v_mfma_f32_16x16x32_f16 a[220:223], v[58:61], v[162:165], a[220:223]
v_mfma_f32_16x16x32_f16 a[220:223], v[62:65], v[166:169], a[220:223]
s_mov_b32 m0, s47
v_mfma_f32_16x16x32_f16 a[224:227], v[86:89], v[162:165], a[224:227]
buffer_load_dwordx4 v13, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[224:227], v[50:53], v[166:169], a[224:227]
v_mfma_f32_16x16x32_f16 a[228:231], v[54:57], v[162:165], a[228:231]
v_mfma_f32_16x16x32_f16 a[228:231], v[110:113], v[166:169], a[228:231]
s_mov_b32 m0, s48
v_mfma_f32_16x16x32_f16 a[232:235], v[114:117], v[162:165], a[232:235]
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[232:235], v[118:121], v[166:169], a[232:235]
v_mfma_f32_16x16x32_f16 a[236:239], v[58:61], v[170:173], a[236:239]
v_mfma_f32_16x16x32_f16 a[236:239], v[62:65], v[174:177], a[236:239]
s_mov_b32 m0, s49
v_mfma_f32_16x16x32_f16 a[240:243], v[86:89], v[170:173], a[240:243]
buffer_load_dwordx4 v15, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[240:243], v[50:53], v[174:177], a[240:243]
v_mfma_f32_16x16x32_f16 a[244:247], v[54:57], v[170:173], a[244:247]
v_mfma_f32_16x16x32_f16 a[244:247], v[110:113], v[174:177], a[244:247]
v_mfma_f32_16x16x32_f16 a[248:251], v[114:117], v[170:173], a[248:251]
s_waitcnt vmcnt(16), lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[248:251], v[118:121], v[174:177], a[248:251]
s_barrier
v_mfma_f32_16x16x32_f16 a[124:127], v[122:125], v[74:77], a[124:127]
v_mfma_f32_16x16x32_f16 a[124:127], v[126:129], v[78:81], a[124:127]
v_mfma_f32_16x16x32_f16 a[0:3], v[130:133], v[74:77], a[0:3]
v_mfma_f32_16x16x32_f16 a[0:3], v[134:137], v[78:81], a[0:3]
v_mfma_f32_16x16x32_f16 a[4:7], v[138:141], v[74:77], a[4:7]
v_mfma_f32_16x16x32_f16 a[4:7], v[142:145], v[78:81], a[4:7]
v_mfma_f32_16x16x32_f16 a[8:11], v[146:149], v[74:77], a[8:11]
v_mfma_f32_16x16x32_f16 a[8:11], v[150:153], v[78:81], a[8:11]
v_mfma_f32_16x16x32_f16 a[12:15], v[122:125], v[82:85], a[12:15]
v_mfma_f32_16x16x32_f16 a[12:15], v[126:129], v[66:69], a[12:15]
v_mfma_f32_16x16x32_f16 a[16:19], v[130:133], v[82:85], a[16:19]
v_mfma_f32_16x16x32_f16 a[16:19], v[134:137], v[66:69], a[16:19]
v_mfma_f32_16x16x32_f16 a[20:23], v[138:141], v[82:85], a[20:23]
v_mfma_f32_16x16x32_f16 a[20:23], v[142:145], v[66:69], a[20:23]
v_mfma_f32_16x16x32_f16 a[24:27], v[146:149], v[82:85], a[24:27]
v_mfma_f32_16x16x32_f16 a[24:27], v[150:153], v[66:69], a[24:27]
v_mfma_f32_16x16x32_f16 a[28:31], v[122:125], v[70:73], a[28:31]
v_mfma_f32_16x16x32_f16 a[28:31], v[126:129], v[90:93], a[28:31]
v_mfma_f32_16x16x32_f16 a[32:35], v[130:133], v[70:73], a[32:35]
v_mfma_f32_16x16x32_f16 a[32:35], v[134:137], v[90:93], a[32:35]
ds_read_b128 v[82:85], v3
v_mfma_f32_16x16x32_f16 a[36:39], v[138:141], v[70:73], a[36:39]
ds_read_b128 v[86:89], v3, offset:64
v_mfma_f32_16x16x32_f16 a[36:39], v[142:145], v[90:93], a[36:39]
ds_read_b128 v[74:77], v3, offset:256
v_mfma_f32_16x16x32_f16 a[40:43], v[146:149], v[70:73], a[40:43]
ds_read_b128 v[78:81], v3, offset:320
v_mfma_f32_16x16x32_f16 a[40:43], v[150:153], v[90:93], a[40:43]
ds_read_b128 v[66:69], v3, offset:512
v_mfma_f32_16x16x32_f16 a[44:47], v[122:125], v[94:97], a[44:47]
ds_read_b128 v[70:73], v3, offset:576
v_mfma_f32_16x16x32_f16 a[44:47], v[126:129], v[98:101], a[44:47]
ds_read_b128 v[58:61], v3, offset:768
v_mfma_f32_16x16x32_f16 a[48:51], v[130:133], v[94:97], a[48:51]
ds_read_b128 v[62:65], v3, offset:832
v_mfma_f32_16x16x32_f16 a[48:51], v[134:137], v[98:101], a[48:51]
ds_read_b128 v[50:53], v3, offset:16896
v_mfma_f32_16x16x32_f16 a[52:55], v[138:141], v[94:97], a[52:55]
ds_read_b128 v[54:57], v3, offset:16960
v_mfma_f32_16x16x32_f16 a[52:55], v[142:145], v[98:101], a[52:55]
ds_read_b128 v[42:45], v3, offset:17152
v_mfma_f32_16x16x32_f16 a[56:59], v[146:149], v[94:97], a[56:59]
ds_read_b128 v[46:49], v3, offset:17216
v_mfma_f32_16x16x32_f16 a[56:59], v[150:153], v[98:101], a[56:59]
ds_read_b128 v[34:37], v3, offset:17408
v_mfma_f32_16x16x32_f16 a[60:63], v[122:125], v[102:105], a[60:63]
ds_read_b128 v[38:41], v3, offset:17472
v_mfma_f32_16x16x32_f16 a[60:63], v[126:129], v[106:109], a[60:63]
ds_read_b128 v[26:29], v3, offset:17664
v_mfma_f32_16x16x32_f16 a[64:67], v[130:133], v[102:105], a[64:67]
ds_read_b128 v[30:33], v3, offset:17728
v_mfma_f32_16x16x32_f16 a[64:67], v[134:137], v[106:109], a[64:67]
ds_read_b128 v[90:93], v25
v_mfma_f32_16x16x32_f16 a[68:71], v[138:141], v[102:105], a[68:71]
ds_read_b128 v[98:101], v25, offset:64
v_mfma_f32_16x16x32_f16 a[68:71], v[142:145], v[106:109], a[68:71]
ds_read_b128 v[94:97], v25, offset:256
v_mfma_f32_16x16x32_f16 a[72:75], v[146:149], v[102:105], a[72:75]
ds_read_b128 v[102:105], v25, offset:320
v_mfma_f32_16x16x32_f16 a[72:75], v[150:153], v[106:109], a[72:75]
ds_read_b128 v[106:109], v25, offset:512
v_mfma_f32_16x16x32_f16 a[76:79], v[122:125], v[154:157], a[76:79]
ds_read_b128 v[110:113], v25, offset:576
v_mfma_f32_16x16x32_f16 a[76:79], v[126:129], v[158:161], a[76:79]
ds_read_b128 v[114:117], v25, offset:768
v_mfma_f32_16x16x32_f16 a[80:83], v[130:133], v[154:157], a[80:83]
ds_read_b128 v[118:121], v25, offset:832
v_mfma_f32_16x16x32_f16 a[80:83], v[134:137], v[158:161], a[80:83]
v_mfma_f32_16x16x32_f16 a[84:87], v[138:141], v[154:157], a[84:87]
v_mfma_f32_16x16x32_f16 a[84:87], v[142:145], v[158:161], a[84:87]
s_mov_b32 m0, s50
v_mfma_f32_16x16x32_f16 a[88:91], v[146:149], v[154:157], a[88:91]
buffer_load_dwordx4 v16, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[88:91], v[150:153], v[158:161], a[88:91]
v_mfma_f32_16x16x32_f16 a[92:95], v[122:125], v[162:165], a[92:95]
v_mfma_f32_16x16x32_f16 a[92:95], v[126:129], v[166:169], a[92:95]
s_mov_b32 m0, s51
v_mfma_f32_16x16x32_f16 a[96:99], v[130:133], v[162:165], a[96:99]
buffer_load_dwordx4 v17, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[96:99], v[134:137], v[166:169], a[96:99]
v_mfma_f32_16x16x32_f16 a[100:103], v[138:141], v[162:165], a[100:103]
v_mfma_f32_16x16x32_f16 a[100:103], v[142:145], v[166:169], a[100:103]
s_mov_b32 m0, s52
v_mfma_f32_16x16x32_f16 a[104:107], v[146:149], v[162:165], a[104:107]
buffer_load_dwordx4 v18, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[104:107], v[150:153], v[166:169], a[104:107]
v_mfma_f32_16x16x32_f16 a[108:111], v[122:125], v[170:173], a[108:111]
v_mfma_f32_16x16x32_f16 a[108:111], v[126:129], v[174:177], a[108:111]
s_mov_b32 m0, s53
v_mfma_f32_16x16x32_f16 a[112:115], v[130:133], v[170:173], a[112:115]
buffer_load_dwordx4 v20, s[16:19], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[112:115], v[134:137], v[174:177], a[112:115]
s_add_u32 s20, s20, 0x100
s_addc_u32 s24, s24, 0
s_add_u32 s10, s10, 0x100
s_addc_u32 s25, s25, 0
s_add_i32 s26, s26, 2
v_mfma_f32_16x16x32_f16 a[116:119], v[138:141], v[170:173], a[116:119]
s_cmpk_lt_u32 s26, 0x7c
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x32_f16 a[116:119], v[142:145], v[174:177], a[116:119]
v_mfma_f32_16x16x32_f16 a[120:123], v[146:149], v[170:173], a[120:123]
v_mfma_f32_16x16x32_f16 a[120:123], v[150:153], v[174:177], a[120:123]
s_cbranch_scc1 .LBB0_1
; %bb.2:
s_waitcnt vmcnt(0), lgkmcnt(0)
s_barrier
v_lshlrev_b32_e32 v9, 3, v21
v_mfma_f32_16x16x32_f16 a[252:255], v[90:93], v[82:85], a[252:255]
v_mfma_f32_16x16x32_f16 a[252:255], v[98:101], v[86:89], a[252:255]
v_mfma_f32_16x16x32_f16 a[128:131], v[94:97], v[82:85], a[128:131]
v_mfma_f32_16x16x32_f16 a[128:131], v[102:105], v[86:89], a[128:131]
v_mfma_f32_16x16x32_f16 a[132:135], v[106:109], v[82:85], a[132:135]
v_mfma_f32_16x16x32_f16 a[132:135], v[110:113], v[86:89], a[132:135]
v_mfma_f32_16x16x32_f16 a[136:139], v[114:117], v[82:85], a[136:139]
v_mfma_f32_16x16x32_f16 a[136:139], v[118:121], v[86:89], a[136:139]
v_mfma_f32_16x16x32_f16 a[140:143], v[90:93], v[74:77], a[140:143]
v_mfma_f32_16x16x32_f16 a[140:143], v[98:101], v[78:81], a[140:143]
v_mfma_f32_16x16x32_f16 a[144:147], v[94:97], v[74:77], a[144:147]
v_mfma_f32_16x16x32_f16 a[144:147], v[102:105], v[78:81], a[144:147]
v_mfma_f32_16x16x32_f16 a[148:151], v[106:109], v[74:77], a[148:151]
v_mfma_f32_16x16x32_f16 a[148:151], v[110:113], v[78:81], a[148:151]
v_mfma_f32_16x16x32_f16 a[152:155], v[114:117], v[74:77], a[152:155]
v_mfma_f32_16x16x32_f16 a[152:155], v[118:121], v[78:81], a[152:155]
v_mfma_f32_16x16x32_f16 a[156:159], v[90:93], v[66:69], a[156:159]
v_mfma_f32_16x16x32_f16 a[156:159], v[98:101], v[70:73], a[156:159]
v_mfma_f32_16x16x32_f16 a[160:163], v[94:97], v[66:69], a[160:163]
v_mfma_f32_16x16x32_f16 a[160:163], v[102:105], v[70:73], a[160:163]
v_mfma_f32_16x16x32_f16 a[164:167], v[106:109], v[66:69], a[164:167]
v_mfma_f32_16x16x32_f16 a[164:167], v[110:113], v[70:73], a[164:167]
v_mfma_f32_16x16x32_f16 a[168:171], v[114:117], v[66:69], a[168:171]
v_mfma_f32_16x16x32_f16 a[168:171], v[118:121], v[70:73], a[168:171]
v_mfma_f32_16x16x32_f16 a[172:175], v[90:93], v[58:61], a[172:175]
v_mfma_f32_16x16x32_f16 a[172:175], v[98:101], v[62:65], a[172:175]
v_mfma_f32_16x16x32_f16 a[176:179], v[94:97], v[58:61], a[176:179]
v_mfma_f32_16x16x32_f16 a[176:179], v[102:105], v[62:65], a[176:179]
v_mfma_f32_16x16x32_f16 a[180:183], v[106:109], v[58:61], a[180:183]
v_mfma_f32_16x16x32_f16 a[180:183], v[110:113], v[62:65], a[180:183]
v_mfma_f32_16x16x32_f16 a[184:187], v[114:117], v[58:61], a[184:187]
v_mfma_f32_16x16x32_f16 a[184:187], v[118:121], v[62:65], a[184:187]
v_mfma_f32_16x16x32_f16 a[188:191], v[90:93], v[50:53], a[188:191]
v_mfma_f32_16x16x32_f16 a[188:191], v[98:101], v[54:57], a[188:191]
v_mfma_f32_16x16x32_f16 a[192:195], v[94:97], v[50:53], a[192:195]
v_mfma_f32_16x16x32_f16 a[192:195], v[102:105], v[54:57], a[192:195]
v_mfma_f32_16x16x32_f16 a[196:199], v[106:109], v[50:53], a[196:199]
v_mfma_f32_16x16x32_f16 a[196:199], v[110:113], v[54:57], a[196:199]
v_mfma_f32_16x16x32_f16 a[200:203], v[114:117], v[50:53], a[200:203]
v_mfma_f32_16x16x32_f16 a[200:203], v[118:121], v[54:57], a[200:203]
v_mfma_f32_16x16x32_f16 a[204:207], v[90:93], v[42:45], a[204:207]
v_mfma_f32_16x16x32_f16 a[204:207], v[98:101], v[46:49], a[204:207]
v_mfma_f32_16x16x32_f16 a[208:211], v[94:97], v[42:45], a[208:211]
v_mfma_f32_16x16x32_f16 a[208:211], v[102:105], v[46:49], a[208:211]
v_mfma_f32_16x16x32_f16 a[212:215], v[106:109], v[42:45], a[212:215]
v_mfma_f32_16x16x32_f16 a[212:215], v[110:113], v[46:49], a[212:215]
v_mfma_f32_16x16x32_f16 a[216:219], v[114:117], v[42:45], a[216:219]
v_mfma_f32_16x16x32_f16 a[216:219], v[118:121], v[46:49], a[216:219]
v_mfma_f32_16x16x32_f16 a[220:223], v[90:93], v[34:37], a[220:223]
v_mfma_f32_16x16x32_f16 a[220:223], v[98:101], v[38:41], a[220:223]
v_mfma_f32_16x16x32_f16 a[224:227], v[94:97], v[34:37], a[224:227]
v_mfma_f32_16x16x32_f16 a[224:227], v[102:105], v[38:41], a[224:227]
v_mfma_f32_16x16x32_f16 a[228:231], v[106:109], v[34:37], a[228:231]
v_mfma_f32_16x16x32_f16 a[228:231], v[110:113], v[38:41], a[228:231]
v_mfma_f32_16x16x32_f16 a[232:235], v[114:117], v[34:37], a[232:235]
v_mfma_f32_16x16x32_f16 a[232:235], v[118:121], v[38:41], a[232:235]
v_mfma_f32_16x16x32_f16 a[236:239], v[90:93], v[26:29], a[236:239]
v_mfma_f32_16x16x32_f16 a[236:239], v[98:101], v[30:33], a[236:239]
v_mfma_f32_16x16x32_f16 a[240:243], v[94:97], v[26:29], a[240:243]
v_mfma_f32_16x16x32_f16 a[240:243], v[102:105], v[30:33], a[240:243]
v_mfma_f32_16x16x32_f16 a[244:247], v[106:109], v[26:29], a[244:247]
v_mfma_f32_16x16x32_f16 a[244:247], v[110:113], v[30:33], a[244:247]
v_mfma_f32_16x16x32_f16 a[248:251], v[114:117], v[26:29], a[248:251]
v_mfma_f32_16x16x32_f16 a[248:251], v[118:121], v[30:33], a[248:251]
v_add_u32_e32 v4, 0x18bc0, v19
ds_read_b128 v[10:13], v4
ds_read_b128 v[14:17], v4, offset:64
ds_read_b128 v[20:23], v4, offset:256
ds_read_b128 v[90:93], v4, offset:320
ds_read_b128 v[94:97], v4, offset:512
ds_read_b128 v[98:101], v4, offset:576
ds_read_b128 v[102:105], v4, offset:768
ds_read_b128 v[4:7], v4, offset:832
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_f16 a[124:127], v[10:13], v[82:85], a[124:127]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_f16 a[124:127], v[14:17], v[86:89], a[124:127]
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_f16 a[0:3], v[20:23], v[82:85], a[0:3]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_f16 a[0:3], v[90:93], v[86:89], a[0:3]
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_f16 a[4:7], v[94:97], v[82:85], a[4:7]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_f16 a[4:7], v[98:101], v[86:89], a[4:7]
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_f16 a[8:11], v[102:105], v[82:85], a[8:11]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[8:11], v[4:7], v[86:89], a[8:11]
v_mfma_f32_16x16x32_f16 a[12:15], v[10:13], v[74:77], a[12:15]
v_mfma_f32_16x16x32_f16 a[12:15], v[14:17], v[78:81], a[12:15]
v_mfma_f32_16x16x32_f16 a[16:19], v[20:23], v[74:77], a[16:19]
v_mfma_f32_16x16x32_f16 a[16:19], v[90:93], v[78:81], a[16:19]
v_mfma_f32_16x16x32_f16 a[20:23], v[94:97], v[74:77], a[20:23]
v_mfma_f32_16x16x32_f16 a[20:23], v[98:101], v[78:81], a[20:23]
v_mfma_f32_16x16x32_f16 a[24:27], v[102:105], v[74:77], a[24:27]
v_mfma_f32_16x16x32_f16 a[24:27], v[4:7], v[78:81], a[24:27]
v_mfma_f32_16x16x32_f16 a[28:31], v[10:13], v[66:69], a[28:31]
v_mfma_f32_16x16x32_f16 a[28:31], v[14:17], v[70:73], a[28:31]
s_nop 7
v_accvgpr_read_b32 v177, a31
v_accvgpr_read_b32 v176, a30
v_accvgpr_read_b32 v175, a29
v_accvgpr_read_b32 v174, a28
v_mfma_f32_16x16x32_f16 a[28:31], v[20:23], v[66:69], a[32:35]
v_mfma_f32_16x16x32_f16 a[28:31], v[90:93], v[70:73], a[28:31]
v_mfma_f32_16x16x32_f16 a[32:35], v[94:97], v[66:69], a[36:39]
v_mfma_f32_16x16x32_f16 a[32:35], v[98:101], v[70:73], a[32:35]
v_mfma_f32_16x16x32_f16 a[36:39], v[102:105], v[66:69], a[40:43]
v_mfma_f32_16x16x32_f16 a[36:39], v[4:7], v[70:73], a[36:39]
v_mfma_f32_16x16x32_f16 a[40:43], v[10:13], v[58:61], a[44:47]
v_mfma_f32_16x16x32_f16 a[40:43], v[14:17], v[62:65], a[40:43]
v_mfma_f32_16x16x32_f16 a[44:47], v[20:23], v[58:61], a[48:51]
v_mfma_f32_16x16x32_f16 a[44:47], v[90:93], v[62:65], a[44:47]
v_mfma_f32_16x16x32_f16 a[48:51], v[94:97], v[58:61], a[52:55]
v_mfma_f32_16x16x32_f16 a[48:51], v[98:101], v[62:65], a[48:51]
v_mfma_f32_16x16x32_f16 a[52:55], v[102:105], v[58:61], a[56:59]
v_mfma_f32_16x16x32_f16 a[52:55], v[4:7], v[62:65], a[52:55]
v_mfma_f32_16x16x32_f16 a[56:59], v[10:13], v[50:53], a[60:63]
v_mfma_f32_16x16x32_f16 a[56:59], v[14:17], v[54:57], a[56:59]
s_nop 7
v_accvgpr_read_b32 v173, a59
v_accvgpr_read_b32 v172, a58
v_accvgpr_read_b32 v171, a57
v_accvgpr_read_b32 v170, a56
v_mfma_f32_16x16x32_f16 a[56:59], v[20:23], v[50:53], a[64:67]
v_mfma_f32_16x16x32_f16 a[64:67], v[90:93], v[54:57], a[56:59]
v_mfma_f32_16x16x32_f16 a[56:59], v[94:97], v[50:53], a[68:71]
v_mfma_f32_16x16x32_f16 a[68:71], v[98:101], v[54:57], a[56:59]
v_mfma_f32_16x16x32_f16 a[56:59], v[102:105], v[50:53], a[72:75]
v_mfma_f32_16x16x32_f16 a[72:75], v[4:7], v[54:57], a[56:59]
v_mfma_f32_16x16x32_f16 a[56:59], v[10:13], v[42:45], a[76:79]
v_mfma_f32_16x16x32_f16 a[76:79], v[14:17], v[46:49], a[56:59]
v_mfma_f32_16x16x32_f16 a[56:59], v[20:23], v[42:45], a[80:83]
v_mfma_f32_16x16x32_f16 a[80:83], v[90:93], v[46:49], a[56:59]
v_mfma_f32_16x16x32_f16 a[56:59], v[94:97], v[42:45], a[84:87]
v_mfma_f32_16x16x32_f16 a[56:59], v[98:101], v[46:49], a[56:59]
v_mfma_f32_16x16x32_f16 a[60:63], v[102:105], v[42:45], a[88:91]
v_mfma_f32_16x16x32_f16 a[60:63], v[4:7], v[46:49], a[60:63]
v_mfma_f32_16x16x32_f16 a[84:87], v[10:13], v[34:37], a[92:95]
v_mfma_f32_16x16x32_f16 a[84:87], v[14:17], v[38:41], a[84:87]
s_nop 7
v_accvgpr_read_b32 v169, a87
v_accvgpr_read_b32 v168, a86
v_accvgpr_read_b32 v167, a85
v_accvgpr_read_b32 v166, a84
v_mfma_f32_16x16x32_f16 a[84:87], v[20:23], v[34:37], a[96:99]
v_mfma_f32_16x16x32_f16 a[96:99], v[90:93], v[38:41], a[84:87]
v_mfma_f32_16x16x32_f16 a[84:87], v[94:97], v[34:37], a[100:103]
v_mfma_f32_16x16x32_f16 a[100:103], v[98:101], v[38:41], a[84:87]
v_mfma_f32_16x16x32_f16 a[84:87], v[102:105], v[34:37], a[104:107]
v_mfma_f32_16x16x32_f16 a[104:107], v[4:7], v[38:41], a[84:87]
v_mfma_f32_16x16x32_f16 a[84:87], v[10:13], v[26:29], a[108:111]
v_mfma_f32_16x16x32_f16 a[108:111], v[14:17], v[30:33], a[84:87]
v_mfma_f32_16x16x32_f16 a[84:87], v[20:23], v[26:29], a[112:115]
v_mfma_f32_16x16x32_f16 a[84:87], v[90:93], v[30:33], a[84:87]
v_mfma_f32_16x16x32_f16 a[88:91], v[94:97], v[26:29], a[116:119]
v_mfma_f32_16x16x32_f16 a[88:91], v[98:101], v[30:33], a[88:91]
v_mfma_f32_16x16x32_f16 a[92:95], v[102:105], v[26:29], a[120:123]
v_mfma_f32_16x16x32_f16 a[92:95], v[4:7], v[30:33], a[92:95]
ds_read_b128 v[118:121], v3, offset:33792
ds_read_b128 v[122:125], v3, offset:33856
ds_read_b128 v[126:129], v3, offset:34048
ds_read_b128 v[130:133], v3, offset:34112
ds_read_b128 v[84:87], v3, offset:34304
ds_read_b128 v[88:91], v3, offset:34368
ds_read_b128 v[76:79], v3, offset:34560
ds_read_b128 v[80:83], v3, offset:34624
ds_read_b128 v[68:71], v3, offset:50688
ds_read_b128 v[72:75], v3, offset:50752
ds_read_b128 v[60:63], v3, offset:50944
ds_read_b128 v[64:67], v3, offset:51008
ds_read_b128 v[20:23], v3, offset:51200
ds_read_b128 v[24:27], v3, offset:51264
ds_read_b128 v[10:13], v3, offset:51456
ds_read_b128 v[14:17], v3, offset:51520
v_add_u32_e32 v3, 0x149e0, v19
ds_read_b128 v[134:137], v3
ds_read_b128 v[138:141], v3, offset:64
ds_read_b128 v[142:145], v3, offset:256
ds_read_b128 v[146:149], v3, offset:320
ds_read_b128 v[150:153], v3, offset:512
ds_read_b128 v[154:157], v3, offset:576
ds_read_b128 v[158:161], v3, offset:768
ds_read_b128 v[162:165], v3, offset:832
v_lshrrev_b32_e32 v3, 4, v1
v_or_b32_e32 v4, 16, v3
v_or_b32_e32 v5, 32, v3
v_or_b32_e32 v6, 48, v3
v_mul_lo_u32 v18, v3, s12
v_mul_lo_u32 v104, v4, s12
v_mul_lo_u32 v105, v5, s12
v_mul_lo_u32 v106, v6, s12
s_mul_i32 s0, s9, s12
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s2, s6, s0
s_addc_u32 s3, s7, s1
s_ashr_i32 s9, s8, 31
s_lshl_b64 s[0:1], s[8:9], 1
s_add_u32 s4, s2, s0
s_addc_u32 s6, s3, s1
s_lshl_b32 s0, s12, 6
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s28, s4, s0
s_addc_u32 s14, s6, s1
s_add_u32 s24, s28, s0
s_addc_u32 s11, s14, s1
s_add_u32 s20, s24, s0
s_addc_u32 s10, s11, s1
s_add_u32 s16, s4, 0x100
s_addc_u32 s9, s6, 0
s_add_u32 s12, s28, 0x100
s_addc_u32 s3, s14, 0
s_add_u32 s8, s24, 0x100
s_addc_u32 s2, s11, 0
s_add_u32 s0, s20, 0x100
s_addc_u32 s1, s10, 0
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_f16 a[112:115], v[134:137], v[118:121], a[252:255]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_f16 a[112:115], v[138:141], v[122:125], a[112:115]
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_f16 a[116:119], v[142:145], v[118:121], a[128:131]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_f16 a[116:119], v[146:149], v[122:125], a[116:119]
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_f16 a[120:123], v[150:153], v[118:121], a[132:135]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_f16 a[120:123], v[154:157], v[122:125], a[120:123]
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_f16 a[128:131], v[158:161], v[118:121], a[136:139]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[128:131], v[162:165], v[122:125], a[128:131]
v_mfma_f32_16x16x32_f16 a[132:135], v[134:137], v[126:129], a[140:143]
v_mfma_f32_16x16x32_f16 a[132:135], v[138:141], v[130:133], a[132:135]
v_mfma_f32_16x16x32_f16 a[136:139], v[142:145], v[126:129], a[144:147]
v_mfma_f32_16x16x32_f16 a[136:139], v[146:149], v[130:133], a[136:139]
v_mfma_f32_16x16x32_f16 a[140:143], v[150:153], v[126:129], a[148:151]
v_mfma_f32_16x16x32_f16 a[140:143], v[154:157], v[130:133], a[140:143]
v_mfma_f32_16x16x32_f16 a[144:147], v[158:161], v[126:129], a[152:155]
v_mfma_f32_16x16x32_f16 a[144:147], v[162:165], v[130:133], a[144:147]
v_add_u32_e32 v3, 0x1cdc0, v19
ds_read_b128 v[28:31], v3
ds_read_b128 v[32:35], v3, offset:64
ds_read_b128 v[36:39], v3, offset:256
ds_read_b128 v[40:43], v3, offset:320
ds_read_b128 v[44:47], v3, offset:512
ds_read_b128 v[48:51], v3, offset:576
ds_read_b128 v[52:55], v3, offset:768
ds_read_b128 v[56:59], v3, offset:832
v_mfma_f32_16x16x32_f16 a[148:151], v[134:137], v[84:87], a[156:159]
v_mfma_f32_16x16x32_f16 a[148:151], v[138:141], v[88:91], a[148:151]
v_mfma_f32_16x16x32_f16 a[152:155], v[142:145], v[84:87], a[160:163]
v_mfma_f32_16x16x32_f16 a[152:155], v[146:149], v[88:91], a[152:155]
v_mfma_f32_16x16x32_f16 a[156:159], v[150:153], v[84:87], a[164:167]
v_mfma_f32_16x16x32_f16 a[156:159], v[154:157], v[88:91], a[156:159]
v_mfma_f32_16x16x32_f16 a[160:163], v[158:161], v[84:87], a[168:171]
v_mfma_f32_16x16x32_f16 a[160:163], v[162:165], v[88:91], a[160:163]
v_mfma_f32_16x16x32_f16 a[164:167], v[134:137], v[76:79], a[172:175]
v_mfma_f32_16x16x32_f16 a[164:167], v[138:141], v[80:83], a[164:167]
v_mfma_f32_16x16x32_f16 a[168:171], v[142:145], v[76:79], a[176:179]
v_mfma_f32_16x16x32_f16 a[168:171], v[146:149], v[80:83], a[168:171]
v_mfma_f32_16x16x32_f16 a[172:175], v[150:153], v[76:79], a[180:183]
v_mfma_f32_16x16x32_f16 a[172:175], v[154:157], v[80:83], a[172:175]
v_mfma_f32_16x16x32_f16 a[176:179], v[158:161], v[76:79], a[184:187]
v_mfma_f32_16x16x32_f16 a[176:179], v[162:165], v[80:83], a[176:179]
v_accvgpr_read_b32 v4, a112
v_accvgpr_read_b32 v3, a113
v_cvt_pk_f16_f32 v4, v4, v3
v_accvgpr_read_b32 v6, a114
v_accvgpr_read_b32 v3, a115
v_cvt_pk_f16_f32 v5, v6, v3
v_accvgpr_read_b32 v6, a116
v_accvgpr_read_b32 v3, a117
v_cvt_pk_f16_f32 v92, v6, v3
v_accvgpr_read_b32 v6, a118
v_accvgpr_read_b32 v3, a119
v_cvt_pk_f16_f32 v93, v6, v3
v_accvgpr_read_b32 v6, a120
v_accvgpr_read_b32 v3, a121
v_cvt_pk_f16_f32 v96, v6, v3
v_accvgpr_read_b32 v6, a122
v_accvgpr_read_b32 v3, a123
v_cvt_pk_f16_f32 v97, v6, v3
v_accvgpr_read_b32 v6, a128
v_accvgpr_read_b32 v3, a129
v_cvt_pk_f16_f32 v100, v6, v3
v_accvgpr_read_b32 v6, a130
v_accvgpr_read_b32 v3, a131
v_cvt_pk_f16_f32 v101, v6, v3
v_accvgpr_read_b32 v6, a132
v_accvgpr_read_b32 v3, a133
v_cvt_pk_f16_f32 v6, v6, v3
v_accvgpr_read_b32 v8, a134
v_accvgpr_read_b32 v3, a135
v_cvt_pk_f16_f32 v7, v8, v3
v_accvgpr_read_b32 v8, a136
v_accvgpr_read_b32 v3, a137
v_cvt_pk_f16_f32 v94, v8, v3
v_accvgpr_read_b32 v8, a138
v_accvgpr_read_b32 v3, a139
v_cvt_pk_f16_f32 v95, v8, v3
v_accvgpr_read_b32 v8, a140
v_accvgpr_read_b32 v3, a141
v_cvt_pk_f16_f32 v98, v8, v3
v_accvgpr_read_b32 v8, a142
v_accvgpr_read_b32 v3, a143
v_cvt_pk_f16_f32 v99, v8, v3
v_accvgpr_read_b32 v8, a144
v_accvgpr_read_b32 v3, a145
v_cvt_pk_f16_f32 v102, v8, v3
v_accvgpr_read_b32 v8, a146
v_accvgpr_read_b32 v3, a147
v_cvt_pk_f16_f32 v103, v8, v3
s_waitcnt lgkmcnt(0)
s_barrier
v_lshlrev_b32_e32 v3, 8, v0
v_and_b32_e32 v8, 0x70, v2
v_and_b32_e32 v19, 1, v0
v_lshlrev_b32_e32 v107, 12, v19
v_and_b32_e32 v108, 16, v0
v_lshlrev_b32_e32 v0, 4, v108
s_lshr_b32 s5, s5, 2
s_and_b32 s7, s13, 0x80
s_movk_i32 s13, 0x2e00
v_and_or_b32 v3, v3, s13, v107
v_or3_b32 v107, s7, v0, v3
v_mov_b32_e32 v0, 0x70
v_bitop3_b32 v109, s5, v2, v0, bitop3:0x78
v_or_b32_e32 v3, v107, v109
v_add_u32_e32 v0, 0, v3
ds_write_b128 v0, v[4:7]
v_xad_u32 v2, v3, 32, 0
ds_write_b128 v2, v[92:95]
v_xad_u32 v3, v3, 64, 0
ds_write_b128 v3, v[96:99]
s_movk_i32 s5, 0x60
v_bitop3_b32 v4, v107, s5, v109, bitop3:0x36
v_add_u32_e32 v4, 0, v4
ds_write_b128 v4, v[100:103]
s_waitcnt lgkmcnt(0)
s_barrier
v_and_b32_e32 v1, 0xe0, v1
v_lshlrev_b32_e32 v5, 4, v1
v_lshrrev_b32_e32 v1, 1, v1
v_lshlrev_b32_e32 v6, 8, v108
v_bitop3_b32 v1, v5, v1, v8, bitop3:0x36
v_lshl_add_u32 v5, v19, 13, 0
v_add3_u32 v1, v5, v6, v1
ds_read_b128 v[92:95], v1
ds_read_b128 v[98:101], v1, offset:256
ds_read_b128 v[108:111], v1, offset:128
ds_read_b128 v[114:117], v1, offset:384
s_and_b32 s5, s6, 0xffff
s_mov_b32 s7, 0x27000
s_mov_b32 s6, 0x7ffffffe
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v96, v92
v_mov_b32_e32 v97, v93
v_add_lshl_u32 v5, v18, v9, 1
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[96:99], v5, s[4:7], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v112, v108
v_mov_b32_e32 v113, v109
v_add_lshl_u32 v6, v104, v9, 1
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[112:115], v6, s[4:7], 0, offen
v_mov_b32_e32 v96, v100
v_mov_b32_e32 v97, v101
v_add_lshl_u32 v7, v105, v9, 1
buffer_store_dwordx4 v[94:97], v7, s[4:7], 0, offen
v_mov_b32_e32 v112, v116
v_mov_b32_e32 v113, v117
v_add_lshl_u32 v8, v106, v9, 1
buffer_store_dwordx4 v[110:113], v8, s[4:7], 0, offen
v_mfma_f32_16x16x32_f16 a[112:115], v[134:137], v[68:71], a[188:191]
v_mfma_f32_16x16x32_f16 a[112:115], v[138:141], v[72:75], a[112:115]
v_mfma_f32_16x16x32_f16 a[116:119], v[142:145], v[68:71], a[192:195]
v_mfma_f32_16x16x32_f16 a[116:119], v[146:149], v[72:75], a[116:119]
v_mfma_f32_16x16x32_f16 a[120:123], v[150:153], v[68:71], a[196:199]
v_mfma_f32_16x16x32_f16 a[120:123], v[154:157], v[72:75], a[120:123]
v_mfma_f32_16x16x32_f16 a[128:131], v[158:161], v[68:71], a[200:203]
v_mfma_f32_16x16x32_f16 a[128:131], v[162:165], v[72:75], a[128:131]
v_mfma_f32_16x16x32_f16 a[132:135], v[134:137], v[60:63], a[204:207]
v_mfma_f32_16x16x32_f16 a[132:135], v[138:141], v[64:67], a[132:135]
v_mfma_f32_16x16x32_f16 a[136:139], v[142:145], v[60:63], a[208:211]
v_mfma_f32_16x16x32_f16 a[136:139], v[146:149], v[64:67], a[136:139]
v_mfma_f32_16x16x32_f16 a[140:143], v[150:153], v[60:63], a[212:215]
v_mfma_f32_16x16x32_f16 a[140:143], v[154:157], v[64:67], a[140:143]
v_mfma_f32_16x16x32_f16 a[144:147], v[158:161], v[60:63], a[216:219]
v_mfma_f32_16x16x32_f16 a[144:147], v[162:165], v[64:67], a[144:147]
v_accvgpr_read_b32 v18, a148
v_accvgpr_read_b32 v9, a149
v_cvt_pk_f16_f32 v92, v18, v9
v_accvgpr_read_b32 v18, a150
v_accvgpr_read_b32 v9, a151
v_cvt_pk_f16_f32 v93, v18, v9
v_accvgpr_read_b32 v18, a152
v_accvgpr_read_b32 v9, a153
v_cvt_pk_f16_f32 v96, v18, v9
v_accvgpr_read_b32 v18, a154
v_accvgpr_read_b32 v9, a155
v_cvt_pk_f16_f32 v97, v18, v9
v_accvgpr_read_b32 v18, a156
v_accvgpr_read_b32 v9, a157
v_cvt_pk_f16_f32 v100, v18, v9
v_accvgpr_read_b32 v18, a158
v_accvgpr_read_b32 v9, a159
v_cvt_pk_f16_f32 v101, v18, v9
v_accvgpr_read_b32 v18, a160
v_accvgpr_read_b32 v9, a161
v_cvt_pk_f16_f32 v104, v18, v9
v_accvgpr_read_b32 v18, a162
v_accvgpr_read_b32 v9, a163
v_cvt_pk_f16_f32 v105, v18, v9
v_accvgpr_read_b32 v18, a164
v_accvgpr_read_b32 v9, a165
v_cvt_pk_f16_f32 v94, v18, v9
v_accvgpr_read_b32 v18, a166
v_accvgpr_read_b32 v9, a167
v_cvt_pk_f16_f32 v95, v18, v9
v_accvgpr_read_b32 v18, a168
v_accvgpr_read_b32 v9, a169
v_cvt_pk_f16_f32 v98, v18, v9
v_accvgpr_read_b32 v18, a170
v_accvgpr_read_b32 v9, a171
v_cvt_pk_f16_f32 v99, v18, v9
v_accvgpr_read_b32 v18, a172
v_accvgpr_read_b32 v9, a173
v_cvt_pk_f16_f32 v102, v18, v9
v_accvgpr_read_b32 v18, a174
v_accvgpr_read_b32 v9, a175
v_cvt_pk_f16_f32 v103, v18, v9
v_accvgpr_read_b32 v18, a176
v_accvgpr_read_b32 v9, a177
v_cvt_pk_f16_f32 v106, v18, v9
v_accvgpr_read_b32 v18, a178
v_accvgpr_read_b32 v9, a179
v_cvt_pk_f16_f32 v107, v18, v9
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[92:95]
ds_write_b128 v2, v[96:99]
ds_write_b128 v3, v[100:103]
ds_write_b128 v4, v[104:107]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[92:95], v1
ds_read_b128 v[98:101], v1, offset:256
ds_read_b128 v[102:105], v1, offset:128
ds_read_b128 v[108:111], v1, offset:384
s_and_b32 s29, s14, 0xffff
s_mov_b32 s30, s6
s_mov_b32 s31, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v96, v92
v_mov_b32_e32 v97, v93
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[96:99], v5, s[28:31], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v106, v102
v_mov_b32_e32 v107, v103
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[106:109], v6, s[28:31], 0, offen
v_mov_b32_e32 v96, v100
v_mov_b32_e32 v97, v101
buffer_store_dwordx4 v[94:97], v7, s[28:31], 0, offen
v_mov_b32_e32 v106, v110
v_mov_b32_e32 v107, v111
buffer_store_dwordx4 v[104:107], v8, s[28:31], 0, offen
v_mfma_f32_16x16x32_f16 a[148:151], v[134:137], v[20:23], a[220:223]
v_mfma_f32_16x16x32_f16 a[148:151], v[138:141], v[24:27], a[148:151]
v_mfma_f32_16x16x32_f16 a[152:155], v[142:145], v[20:23], a[224:227]
v_mfma_f32_16x16x32_f16 a[152:155], v[146:149], v[24:27], a[152:155]
v_mfma_f32_16x16x32_f16 a[156:159], v[150:153], v[20:23], a[228:231]
v_mfma_f32_16x16x32_f16 a[156:159], v[154:157], v[24:27], a[156:159]
v_mfma_f32_16x16x32_f16 a[160:163], v[158:161], v[20:23], a[232:235]
v_mfma_f32_16x16x32_f16 a[160:163], v[162:165], v[24:27], a[160:163]
v_mfma_f32_16x16x32_f16 a[164:167], v[134:137], v[10:13], a[236:239]
v_mfma_f32_16x16x32_f16 a[164:167], v[138:141], v[14:17], a[164:167]
v_mfma_f32_16x16x32_f16 a[168:171], v[142:145], v[10:13], a[240:243]
v_mfma_f32_16x16x32_f16 a[168:171], v[146:149], v[14:17], a[168:171]
v_mfma_f32_16x16x32_f16 a[172:175], v[150:153], v[10:13], a[244:247]
v_mfma_f32_16x16x32_f16 a[172:175], v[154:157], v[14:17], a[172:175]
v_mfma_f32_16x16x32_f16 a[176:179], v[158:161], v[10:13], a[248:251]
v_mfma_f32_16x16x32_f16 a[176:179], v[162:165], v[14:17], a[176:179]
v_accvgpr_read_b32 v18, a112
v_accvgpr_read_b32 v9, a113
v_cvt_pk_f16_f32 v92, v18, v9
v_accvgpr_read_b32 v18, a114
v_accvgpr_read_b32 v9, a115
v_cvt_pk_f16_f32 v93, v18, v9
v_accvgpr_read_b32 v18, a116
v_accvgpr_read_b32 v9, a117
v_cvt_pk_f16_f32 v96, v18, v9
v_accvgpr_read_b32 v18, a118
v_accvgpr_read_b32 v9, a119
v_cvt_pk_f16_f32 v97, v18, v9
v_accvgpr_read_b32 v18, a120
v_accvgpr_read_b32 v9, a121
v_cvt_pk_f16_f32 v100, v18, v9
v_accvgpr_read_b32 v18, a122
v_accvgpr_read_b32 v9, a123
v_cvt_pk_f16_f32 v101, v18, v9
v_accvgpr_read_b32 v18, a128
v_accvgpr_read_b32 v9, a129
v_cvt_pk_f16_f32 v104, v18, v9
v_accvgpr_read_b32 v18, a130
v_accvgpr_read_b32 v9, a131
v_cvt_pk_f16_f32 v105, v18, v9
v_accvgpr_read_b32 v18, a132
v_accvgpr_read_b32 v9, a133
v_cvt_pk_f16_f32 v94, v18, v9
v_accvgpr_read_b32 v18, a134
v_accvgpr_read_b32 v9, a135
v_cvt_pk_f16_f32 v95, v18, v9
v_accvgpr_read_b32 v18, a136
v_accvgpr_read_b32 v9, a137
v_cvt_pk_f16_f32 v98, v18, v9
v_accvgpr_read_b32 v18, a138
v_accvgpr_read_b32 v9, a139
v_cvt_pk_f16_f32 v99, v18, v9
v_accvgpr_read_b32 v18, a140
v_accvgpr_read_b32 v9, a141
v_cvt_pk_f16_f32 v102, v18, v9
v_accvgpr_read_b32 v18, a142
v_accvgpr_read_b32 v9, a143
v_cvt_pk_f16_f32 v103, v18, v9
v_accvgpr_read_b32 v18, a144
v_accvgpr_read_b32 v9, a145
v_cvt_pk_f16_f32 v106, v18, v9
v_accvgpr_read_b32 v18, a146
v_accvgpr_read_b32 v9, a147
v_cvt_pk_f16_f32 v107, v18, v9
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[92:95]
ds_write_b128 v2, v[96:99]
ds_write_b128 v3, v[100:103]
ds_write_b128 v4, v[104:107]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[92:95], v1
ds_read_b128 v[98:101], v1, offset:256
ds_read_b128 v[102:105], v1, offset:128
ds_read_b128 v[108:111], v1, offset:384
s_and_b32 s25, s11, 0xffff
s_mov_b32 s26, s6
s_mov_b32 s27, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v96, v92
v_mov_b32_e32 v97, v93
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[96:99], v5, s[24:27], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v106, v102
v_mov_b32_e32 v107, v103
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[106:109], v6, s[24:27], 0, offen
v_mov_b32_e32 v96, v100
v_mov_b32_e32 v97, v101
buffer_store_dwordx4 v[94:97], v7, s[24:27], 0, offen
v_mov_b32_e32 v106, v110
v_mov_b32_e32 v107, v111
buffer_store_dwordx4 v[104:107], v8, s[24:27], 0, offen
v_mfma_f32_16x16x32_f16 a[112:115], v[28:31], v[118:121], a[124:127]
v_mfma_f32_16x16x32_f16 a[112:115], v[32:35], v[122:125], a[112:115]
v_mfma_f32_16x16x32_f16 a[0:3], v[36:39], v[118:121], a[0:3]
v_mfma_f32_16x16x32_f16 a[0:3], v[40:43], v[122:125], a[0:3]
v_mfma_f32_16x16x32_f16 a[4:7], v[44:47], v[118:121], a[4:7]
v_mfma_f32_16x16x32_f16 a[4:7], v[48:51], v[122:125], a[4:7]
v_mfma_f32_16x16x32_f16 a[8:11], v[52:55], v[118:121], a[8:11]
v_mfma_f32_16x16x32_f16 a[8:11], v[56:59], v[122:125], a[8:11]
v_mfma_f32_16x16x32_f16 a[12:15], v[28:31], v[126:129], a[12:15]
v_mfma_f32_16x16x32_f16 a[12:15], v[32:35], v[130:133], a[12:15]
v_mfma_f32_16x16x32_f16 a[16:19], v[36:39], v[126:129], a[16:19]
v_mfma_f32_16x16x32_f16 a[16:19], v[40:43], v[130:133], a[16:19]
v_mfma_f32_16x16x32_f16 a[20:23], v[44:47], v[126:129], a[20:23]
v_mfma_f32_16x16x32_f16 a[20:23], v[48:51], v[130:133], a[20:23]
v_mfma_f32_16x16x32_f16 a[24:27], v[52:55], v[126:129], a[24:27]
v_mfma_f32_16x16x32_f16 a[24:27], v[56:59], v[130:133], a[24:27]
v_accvgpr_read_b32 v18, a148
v_accvgpr_read_b32 v9, a149
v_cvt_pk_f16_f32 v92, v18, v9
v_accvgpr_read_b32 v18, a150
v_accvgpr_read_b32 v9, a151
v_cvt_pk_f16_f32 v93, v18, v9
v_accvgpr_read_b32 v18, a152
v_accvgpr_read_b32 v9, a153
v_cvt_pk_f16_f32 v96, v18, v9
v_accvgpr_read_b32 v18, a154
v_accvgpr_read_b32 v9, a155
v_cvt_pk_f16_f32 v97, v18, v9
v_accvgpr_read_b32 v18, a156
v_accvgpr_read_b32 v9, a157
v_cvt_pk_f16_f32 v100, v18, v9
v_accvgpr_read_b32 v18, a158
v_accvgpr_read_b32 v9, a159
v_cvt_pk_f16_f32 v101, v18, v9
v_accvgpr_read_b32 v18, a160
v_accvgpr_read_b32 v9, a161
v_cvt_pk_f16_f32 v104, v18, v9
v_accvgpr_read_b32 v18, a162
v_accvgpr_read_b32 v9, a163
v_cvt_pk_f16_f32 v105, v18, v9
v_accvgpr_read_b32 v18, a164
v_accvgpr_read_b32 v9, a165
v_cvt_pk_f16_f32 v94, v18, v9
v_accvgpr_read_b32 v18, a166
v_accvgpr_read_b32 v9, a167
v_cvt_pk_f16_f32 v95, v18, v9
v_accvgpr_read_b32 v18, a168
v_accvgpr_read_b32 v9, a169
v_cvt_pk_f16_f32 v98, v18, v9
v_accvgpr_read_b32 v18, a170
v_accvgpr_read_b32 v9, a171
v_cvt_pk_f16_f32 v99, v18, v9
v_accvgpr_read_b32 v18, a172
v_accvgpr_read_b32 v9, a173
v_cvt_pk_f16_f32 v102, v18, v9
v_accvgpr_read_b32 v18, a174
v_accvgpr_read_b32 v9, a175
v_cvt_pk_f16_f32 v103, v18, v9
v_accvgpr_read_b32 v18, a176
v_accvgpr_read_b32 v9, a177
v_cvt_pk_f16_f32 v106, v18, v9
v_accvgpr_read_b32 v18, a178
v_accvgpr_read_b32 v9, a179
v_cvt_pk_f16_f32 v107, v18, v9
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[92:95]
ds_write_b128 v2, v[96:99]
ds_write_b128 v3, v[100:103]
ds_write_b128 v4, v[104:107]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[92:95], v1
ds_read_b128 v[98:101], v1, offset:256
ds_read_b128 v[102:105], v1, offset:128
ds_read_b128 v[108:111], v1, offset:384
s_and_b32 s21, s10, 0xffff
s_mov_b32 s22, s6
s_mov_b32 s23, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v96, v92
v_mov_b32_e32 v97, v93
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[96:99], v5, s[20:23], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v106, v102
v_mov_b32_e32 v107, v103
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[106:109], v6, s[20:23], 0, offen
v_mov_b32_e32 v96, v100
v_mov_b32_e32 v97, v101
buffer_store_dwordx4 v[94:97], v7, s[20:23], 0, offen
v_mov_b32_e32 v106, v110
v_mov_b32_e32 v107, v111
buffer_store_dwordx4 v[104:107], v8, s[20:23], 0, offen
v_accvgpr_write_b32 a116, v174
v_accvgpr_write_b32 a117, v175
v_accvgpr_write_b32 a118, v176
v_accvgpr_write_b32 a119, v177
s_nop 1
v_mfma_f32_16x16x32_f16 a[116:119], v[28:31], v[84:87], a[116:119]
v_mfma_f32_16x16x32_f16 a[116:119], v[32:35], v[88:91], a[116:119]
v_mfma_f32_16x16x32_f16 a[28:31], v[36:39], v[84:87], a[28:31]
v_mfma_f32_16x16x32_f16 a[28:31], v[40:43], v[88:91], a[28:31]
v_mfma_f32_16x16x32_f16 a[32:35], v[44:47], v[84:87], a[32:35]
v_mfma_f32_16x16x32_f16 a[32:35], v[48:51], v[88:91], a[32:35]
v_mfma_f32_16x16x32_f16 a[36:39], v[52:55], v[84:87], a[36:39]
v_mfma_f32_16x16x32_f16 a[36:39], v[56:59], v[88:91], a[36:39]
v_mfma_f32_16x16x32_f16 a[40:43], v[28:31], v[76:79], a[40:43]
v_mfma_f32_16x16x32_f16 a[40:43], v[32:35], v[80:83], a[40:43]
v_mfma_f32_16x16x32_f16 a[44:47], v[36:39], v[76:79], a[44:47]
v_mfma_f32_16x16x32_f16 a[44:47], v[40:43], v[80:83], a[44:47]
v_mfma_f32_16x16x32_f16 a[48:51], v[44:47], v[76:79], a[48:51]
v_mfma_f32_16x16x32_f16 a[48:51], v[48:51], v[80:83], a[48:51]
v_mfma_f32_16x16x32_f16 a[52:55], v[52:55], v[76:79], a[52:55]
v_mfma_f32_16x16x32_f16 a[52:55], v[56:59], v[80:83], a[52:55]
v_accvgpr_read_b32 v18, a112
v_accvgpr_read_b32 v9, a113
v_cvt_pk_f16_f32 v76, v18, v9
v_accvgpr_read_b32 v18, a114
v_accvgpr_read_b32 v9, a115
v_cvt_pk_f16_f32 v77, v18, v9
v_accvgpr_read_b32 v18, a0
v_accvgpr_read_b32 v9, a1
v_cvt_pk_f16_f32 v80, v18, v9
v_accvgpr_read_b32 v18, a2
v_accvgpr_read_b32 v9, a3
v_cvt_pk_f16_f32 v81, v18, v9
v_accvgpr_read_b32 v18, a4
v_accvgpr_read_b32 v9, a5
v_cvt_pk_f16_f32 v84, v18, v9
v_accvgpr_read_b32 v18, a6
v_accvgpr_read_b32 v9, a7
v_cvt_pk_f16_f32 v85, v18, v9
v_accvgpr_read_b32 v18, a8
v_accvgpr_read_b32 v9, a9
v_cvt_pk_f16_f32 v88, v18, v9
v_accvgpr_read_b32 v18, a10
v_accvgpr_read_b32 v9, a11
v_cvt_pk_f16_f32 v89, v18, v9
v_accvgpr_read_b32 v18, a12
v_accvgpr_read_b32 v9, a13
v_cvt_pk_f16_f32 v78, v18, v9
v_accvgpr_read_b32 v18, a14
v_accvgpr_read_b32 v9, a15
v_cvt_pk_f16_f32 v79, v18, v9
v_accvgpr_read_b32 v18, a16
v_accvgpr_read_b32 v9, a17
v_cvt_pk_f16_f32 v82, v18, v9
v_accvgpr_read_b32 v18, a18
v_accvgpr_read_b32 v9, a19
v_cvt_pk_f16_f32 v83, v18, v9
v_accvgpr_read_b32 v18, a20
v_accvgpr_read_b32 v9, a21
v_cvt_pk_f16_f32 v86, v18, v9
v_accvgpr_read_b32 v18, a22
v_accvgpr_read_b32 v9, a23
v_cvt_pk_f16_f32 v87, v18, v9
v_accvgpr_read_b32 v18, a24
v_accvgpr_read_b32 v9, a25
v_cvt_pk_f16_f32 v90, v18, v9
v_accvgpr_read_b32 v18, a26
v_accvgpr_read_b32 v9, a27
v_cvt_pk_f16_f32 v91, v18, v9
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[76:79]
ds_write_b128 v2, v[80:83]
ds_write_b128 v3, v[84:87]
ds_write_b128 v4, v[88:91]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[76:79], v1
ds_read_b128 v[82:85], v1, offset:256
ds_read_b128 v[86:89], v1, offset:128
ds_read_b128 v[92:95], v1, offset:384
s_and_b32 s17, s9, 0xffff
s_mov_b32 s18, s6
s_mov_b32 s19, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v80, v76
v_mov_b32_e32 v81, v77
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[80:83], v5, s[16:19], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v90, v86
v_mov_b32_e32 v91, v87
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[90:93], v6, s[16:19], 0, offen
v_mov_b32_e32 v80, v84
v_mov_b32_e32 v81, v85
buffer_store_dwordx4 v[78:81], v7, s[16:19], 0, offen
v_mov_b32_e32 v90, v94
v_mov_b32_e32 v91, v95
buffer_store_dwordx4 v[88:91], v8, s[16:19], 0, offen
v_accvgpr_write_b32 a0, v170
v_accvgpr_write_b32 a1, v171
v_accvgpr_write_b32 a2, v172
v_accvgpr_write_b32 a3, v173
s_nop 1
v_mfma_f32_16x16x32_f16 a[0:3], v[28:31], v[68:71], a[0:3]
v_mfma_f32_16x16x32_f16 a[0:3], v[32:35], v[72:75], a[0:3]
v_mfma_f32_16x16x32_f16 a[4:7], v[36:39], v[68:71], a[64:67]
v_mfma_f32_16x16x32_f16 a[4:7], v[40:43], v[72:75], a[4:7]
v_mfma_f32_16x16x32_f16 a[8:11], v[44:47], v[68:71], a[68:71]
v_mfma_f32_16x16x32_f16 a[8:11], v[48:51], v[72:75], a[8:11]
v_mfma_f32_16x16x32_f16 a[12:15], v[52:55], v[68:71], a[72:75]
v_mfma_f32_16x16x32_f16 a[12:15], v[56:59], v[72:75], a[12:15]
v_mfma_f32_16x16x32_f16 a[16:19], v[28:31], v[60:63], a[76:79]
v_mfma_f32_16x16x32_f16 a[16:19], v[32:35], v[64:67], a[16:19]
v_mfma_f32_16x16x32_f16 a[20:23], v[36:39], v[60:63], a[80:83]
v_mfma_f32_16x16x32_f16 a[20:23], v[40:43], v[64:67], a[20:23]
v_mfma_f32_16x16x32_f16 a[24:27], v[44:47], v[60:63], a[56:59]
v_mfma_f32_16x16x32_f16 a[24:27], v[48:51], v[64:67], a[24:27]
v_mfma_f32_16x16x32_f16 a[56:59], v[52:55], v[60:63], a[60:63]
v_mfma_f32_16x16x32_f16 a[56:59], v[56:59], v[64:67], a[56:59]
v_accvgpr_read_b32 v18, a116
v_accvgpr_read_b32 v9, a117
v_cvt_pk_f16_f32 v60, v18, v9
v_accvgpr_read_b32 v18, a118
v_accvgpr_read_b32 v9, a119
v_cvt_pk_f16_f32 v61, v18, v9
v_accvgpr_read_b32 v18, a28
v_accvgpr_read_b32 v9, a29
v_cvt_pk_f16_f32 v64, v18, v9
v_accvgpr_read_b32 v18, a30
v_accvgpr_read_b32 v9, a31
v_cvt_pk_f16_f32 v65, v18, v9
v_accvgpr_read_b32 v18, a32
v_accvgpr_read_b32 v9, a33
v_cvt_pk_f16_f32 v68, v18, v9
v_accvgpr_read_b32 v18, a34
v_accvgpr_read_b32 v9, a35
v_cvt_pk_f16_f32 v69, v18, v9
v_accvgpr_read_b32 v18, a36
v_accvgpr_read_b32 v9, a37
v_cvt_pk_f16_f32 v72, v18, v9
v_accvgpr_read_b32 v18, a38
v_accvgpr_read_b32 v9, a39
v_cvt_pk_f16_f32 v73, v18, v9
v_accvgpr_read_b32 v18, a40
v_accvgpr_read_b32 v9, a41
v_cvt_pk_f16_f32 v62, v18, v9
v_accvgpr_read_b32 v18, a42
v_accvgpr_read_b32 v9, a43
v_cvt_pk_f16_f32 v63, v18, v9
v_accvgpr_read_b32 v18, a44
v_accvgpr_read_b32 v9, a45
v_cvt_pk_f16_f32 v66, v18, v9
v_accvgpr_read_b32 v18, a46
v_accvgpr_read_b32 v9, a47
v_cvt_pk_f16_f32 v67, v18, v9
v_accvgpr_read_b32 v18, a48
v_accvgpr_read_b32 v9, a49
v_cvt_pk_f16_f32 v70, v18, v9
v_accvgpr_read_b32 v18, a50
v_accvgpr_read_b32 v9, a51
v_cvt_pk_f16_f32 v71, v18, v9
v_accvgpr_read_b32 v18, a52
v_accvgpr_read_b32 v9, a53
v_cvt_pk_f16_f32 v74, v18, v9
v_accvgpr_read_b32 v18, a54
v_accvgpr_read_b32 v9, a55
v_cvt_pk_f16_f32 v75, v18, v9
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[60:63]
ds_write_b128 v2, v[64:67]
ds_write_b128 v3, v[68:71]
ds_write_b128 v4, v[72:75]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[60:63], v1
ds_read_b128 v[66:69], v1, offset:256
ds_read_b128 v[70:73], v1, offset:128
ds_read_b128 v[76:79], v1, offset:384
s_and_b32 s13, s3, 0xffff
s_mov_b32 s14, s6
s_mov_b32 s15, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v64, v60
v_mov_b32_e32 v65, v61
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[64:67], v5, s[12:15], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v74, v70
v_mov_b32_e32 v75, v71
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[74:77], v6, s[12:15], 0, offen
v_mov_b32_e32 v64, v68
v_mov_b32_e32 v65, v69
buffer_store_dwordx4 v[62:65], v7, s[12:15], 0, offen
v_mov_b32_e32 v74, v78
v_mov_b32_e32 v75, v79
buffer_store_dwordx4 v[72:75], v8, s[12:15], 0, offen
v_accvgpr_write_b32 a28, v166
v_accvgpr_write_b32 a29, v167
v_accvgpr_write_b32 a30, v168
v_accvgpr_write_b32 a31, v169
s_nop 1
v_mfma_f32_16x16x32_f16 a[28:31], v[28:31], v[20:23], a[28:31]
v_mfma_f32_16x16x32_f16 a[28:31], v[32:35], v[24:27], a[28:31]
v_mfma_f32_16x16x32_f16 a[32:35], v[36:39], v[20:23], a[96:99]
v_mfma_f32_16x16x32_f16 a[32:35], v[40:43], v[24:27], a[32:35]
v_mfma_f32_16x16x32_f16 a[36:39], v[44:47], v[20:23], a[100:103]
v_mfma_f32_16x16x32_f16 a[36:39], v[48:51], v[24:27], a[36:39]
v_mfma_f32_16x16x32_f16 a[40:43], v[52:55], v[20:23], a[104:107]
v_mfma_f32_16x16x32_f16 a[40:43], v[56:59], v[24:27], a[40:43]
v_mfma_f32_16x16x32_f16 a[44:47], v[28:31], v[10:13], a[108:111]
v_mfma_f32_16x16x32_f16 a[44:47], v[32:35], v[14:17], a[44:47]
v_mfma_f32_16x16x32_f16 a[48:51], v[36:39], v[10:13], a[84:87]
v_mfma_f32_16x16x32_f16 a[48:51], v[40:43], v[14:17], a[48:51]
v_mfma_f32_16x16x32_f16 a[52:55], v[44:47], v[10:13], a[88:91]
v_mfma_f32_16x16x32_f16 a[52:55], v[48:51], v[14:17], a[52:55]
v_mfma_f32_16x16x32_f16 a[60:63], v[52:55], v[10:13], a[92:95]
v_mfma_f32_16x16x32_f16 a[60:63], v[56:59], v[14:17], a[60:63]
v_accvgpr_read_b32 v10, a0
v_accvgpr_read_b32 v9, a1
v_cvt_pk_f16_f32 v10, v10, v9
v_accvgpr_read_b32 v12, a2
v_accvgpr_read_b32 v9, a3
v_cvt_pk_f16_f32 v11, v12, v9
v_accvgpr_read_b32 v12, a4
v_accvgpr_read_b32 v9, a5
v_cvt_pk_f16_f32 v14, v12, v9
v_accvgpr_read_b32 v12, a6
v_accvgpr_read_b32 v9, a7
v_cvt_pk_f16_f32 v15, v12, v9
v_accvgpr_read_b32 v12, a8
v_accvgpr_read_b32 v9, a9
v_cvt_pk_f16_f32 v18, v12, v9
v_accvgpr_read_b32 v12, a10
v_accvgpr_read_b32 v9, a11
v_cvt_pk_f16_f32 v19, v12, v9
v_accvgpr_read_b32 v12, a12
v_accvgpr_read_b32 v9, a13
v_cvt_pk_f16_f32 v22, v12, v9
v_accvgpr_read_b32 v12, a14
v_accvgpr_read_b32 v9, a15
v_cvt_pk_f16_f32 v23, v12, v9
v_accvgpr_read_b32 v12, a16
v_accvgpr_read_b32 v9, a17
v_cvt_pk_f16_f32 v12, v12, v9
v_accvgpr_read_b32 v16, a18
v_accvgpr_read_b32 v9, a19
v_cvt_pk_f16_f32 v13, v16, v9
v_accvgpr_read_b32 v16, a20
v_accvgpr_read_b32 v9, a21
v_cvt_pk_f16_f32 v16, v16, v9
v_accvgpr_read_b32 v20, a22
v_accvgpr_read_b32 v9, a23
v_cvt_pk_f16_f32 v17, v20, v9
v_accvgpr_read_b32 v20, a24
v_accvgpr_read_b32 v9, a25
v_cvt_pk_f16_f32 v20, v20, v9
v_accvgpr_read_b32 v24, a26
v_accvgpr_read_b32 v9, a27
v_cvt_pk_f16_f32 v21, v24, v9
v_accvgpr_read_b32 v24, a56
v_accvgpr_read_b32 v9, a57
v_cvt_pk_f16_f32 v24, v24, v9
v_accvgpr_read_b32 v26, a58
v_accvgpr_read_b32 v9, a59
v_cvt_pk_f16_f32 v25, v26, v9
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[10:13]
ds_write_b128 v2, v[14:17]
ds_write_b128 v3, v[18:21]
ds_write_b128 v4, v[22:25]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[10:13], v1
ds_read_b128 v[16:19], v1, offset:256
ds_read_b128 v[20:23], v1, offset:128
ds_read_b128 v[26:29], v1, offset:384
s_and_b32 s9, s2, 0xffff
s_mov_b32 s10, s6
s_mov_b32 s11, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v14, v10
v_mov_b32_e32 v15, v11
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[14:17], v5, s[8:11], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v24, v20
v_mov_b32_e32 v25, v21
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[24:27], v6, s[8:11], 0, offen
v_mov_b32_e32 v14, v18
v_mov_b32_e32 v15, v19
buffer_store_dwordx4 v[12:15], v7, s[8:11], 0, offen
v_mov_b32_e32 v24, v28
v_mov_b32_e32 v25, v29
buffer_store_dwordx4 v[22:25], v8, s[8:11], 0, offen
v_accvgpr_read_b32 v10, a28
v_accvgpr_read_b32 v9, a29
v_cvt_pk_f16_f32 v10, v10, v9
v_accvgpr_read_b32 v12, a30
v_accvgpr_read_b32 v9, a31
v_cvt_pk_f16_f32 v11, v12, v9
v_accvgpr_read_b32 v12, a32
v_accvgpr_read_b32 v9, a33
v_cvt_pk_f16_f32 v14, v12, v9
v_accvgpr_read_b32 v12, a34
v_accvgpr_read_b32 v9, a35
v_cvt_pk_f16_f32 v15, v12, v9
v_accvgpr_read_b32 v12, a36
v_accvgpr_read_b32 v9, a37
v_cvt_pk_f16_f32 v18, v12, v9
v_accvgpr_read_b32 v12, a38
v_accvgpr_read_b32 v9, a39
v_cvt_pk_f16_f32 v19, v12, v9
v_accvgpr_read_b32 v12, a40
v_accvgpr_read_b32 v9, a41
v_cvt_pk_f16_f32 v22, v12, v9
v_accvgpr_read_b32 v12, a42
v_accvgpr_read_b32 v9, a43
v_cvt_pk_f16_f32 v23, v12, v9
v_accvgpr_read_b32 v12, a44
v_accvgpr_read_b32 v9, a45
v_cvt_pk_f16_f32 v12, v12, v9
v_accvgpr_read_b32 v16, a46
v_accvgpr_read_b32 v9, a47
v_cvt_pk_f16_f32 v13, v16, v9
v_accvgpr_read_b32 v16, a48
v_accvgpr_read_b32 v9, a49
v_cvt_pk_f16_f32 v16, v16, v9
v_accvgpr_read_b32 v20, a50
v_accvgpr_read_b32 v9, a51
v_cvt_pk_f16_f32 v17, v20, v9
v_accvgpr_read_b32 v20, a52
v_accvgpr_read_b32 v9, a53
v_cvt_pk_f16_f32 v20, v20, v9
v_accvgpr_read_b32 v24, a54
v_accvgpr_read_b32 v9, a55
v_cvt_pk_f16_f32 v21, v24, v9
v_accvgpr_read_b32 v24, a60
v_accvgpr_read_b32 v9, a61
v_cvt_pk_f16_f32 v24, v24, v9
v_accvgpr_read_b32 v26, a62
v_accvgpr_read_b32 v9, a63
v_cvt_pk_f16_f32 v25, v26, v9
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[10:13]
ds_write_b128 v2, v[14:17]
ds_write_b128 v3, v[18:21]
ds_write_b128 v4, v[22:25]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[10:13], v1
ds_read_b128 v[16:19], v1, offset:256
ds_read_b128 v[20:23], v1, offset:128
ds_read_b128 v[26:29], v1, offset:384
s_and_b32 s1, s1, 0xffff
s_mov_b32 s2, s6
s_mov_b32 s3, s7
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v14, v10
v_mov_b32_e32 v15, v11
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[14:17], v5, s[0:3], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v24, v20
v_mov_b32_e32 v25, v21
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[24:27], v6, s[0:3], 0, offen
v_mov_b32_e32 v14, v18
v_mov_b32_e32 v15, v19
buffer_store_dwordx4 v[12:15], v7, s[0:3], 0, offen
v_mov_b32_e32 v24, v28
v_mov_b32_e32 v25, v29
buffer_store_dwordx4 v[22:25], v8, s[0:3], 0, offen
s_endpgm
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v8_beyond_hotloop
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 55
		.amdhsa_accum_offset 180
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	v8_beyond_hotloop, .Lfunc_end0-v8_beyond_hotloop
	.cfi_endproc
                                        ; -- End function
	.set v8_beyond_hotloop.num_vgpr, 178
	.set v8_beyond_hotloop.num_agpr, 256
	.set v8_beyond_hotloop.numbered_sgpr, 55
	.set v8_beyond_hotloop.num_named_barrier, 0
	.set v8_beyond_hotloop.private_seg_size, 0
	.set v8_beyond_hotloop.uses_vcc, 0
	.set v8_beyond_hotloop.uses_flat_scratch, 0
	.set v8_beyond_hotloop.has_dyn_sized_stack, 0
	.set v8_beyond_hotloop.has_recursion, 0
	.set v8_beyond_hotloop.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14972
; TotalNumSgprs: 61
; NumVgprs: 178
; NumAgprs: 256
; TotalNumVgprs: 436
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 54
; NumSGPRsForWavesPerEU: 61
; NumVGPRsForWavesPerEU: 436
; AccumOffset: 180
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 44
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7 ; matmul_kernel.py
.Linfo_string2:
	.asciz	"/var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v8_beyond_hotloop" ; string offset=24 ; /var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v8_beyond_hotloop
.Linfo_string3:
	.asciz	"v8_beyond_hotloop"             ; string offset=100 ; v8_beyond_hotloop
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .max_flat_workgroup_size: 256
    .name:           v8_beyond_hotloop
    .private_segment_fixed_size: 0
    .sgpr_count:     61
    .sgpr_spill_count: 0
    .symbol:         v8_beyond_hotloop.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 512
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0: