	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v7_slice                        ; -- Begin function v7_slice
	.p2align	8
	.type	v7_slice,@function
v7_slice:                               ; @v7_slice
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
s_bfe_u32 s17, s13, 0x20006
s_add_i32 s0, s9, 0xff
s_ashr_i32 s1, s0, 31
s_lshr_b32 s1, s1, 24
s_add_i32 s0, s0, s1
s_ashr_i32 s0, s0, 8
s_xor_b32 s1, s16, s0
s_ashr_i32 s1, s1, 31
s_abs_i32 s8, s16
s_abs_i32 s9, s0
v_cvt_f32_u32_e32 v1, s9
v_rcp_iflag_f32_e32 v1, v1
s_nop 0
v_mul_f32_e32 v1, 0x4f7ffffe, v1
v_cvt_u32_f32_e32 v1, v1
s_sub_i32 s14, 0, s9
v_readfirstlane_b32 s15, v1
s_mul_i32 s14, s14, s15
s_mul_hi_u32 s14, s15, s14
s_add_i32 s15, s15, s14
s_mul_hi_u32 s14, s8, s15
s_mul_i32 s15, s14, s9
s_sub_i32 s8, s8, s15
s_add_i32 s15, s14, 1
s_sub_i32 s18, s8, s9
s_cmp_ge_u32 s8, s9
s_cselect_b32 s14, s15, s14
s_cselect_b32 s8, s18, s8
s_add_i32 s15, s14, 1
s_cmp_ge_u32 s8, s9
s_cselect_b32 s8, s15, s14
s_xor_b32 s8, s8, s1
s_sub_i32 s1, s8, s1
s_mul_i32 s0, s1, s0
s_sub_i32 s8, s16, s0
v_and_b32_e32 v1, 63, v0
v_lshl_or_b32 v1, s17, 6, v1
v_lshlrev_b32_e32 v2, 1, v0
v_and_b32_e32 v2, 0x70, v2
v_or_b32_e32 v3, s17, v2
v_or_b32_e32 v4, 4, v3
v_or_b32_e32 v5, 8, v3
v_or_b32_e32 v7, 12, v3
v_or_b32_e32 v8, 0x80, v3
v_or_b32_e32 v9, 0x84, v3
v_or_b32_e32 v10, 0x88, v3
v_or_b32_e32 v11, 0x8c, v3
v_lshlrev_b32_e32 v2, 3, v0
v_and_b32_e32 v6, 56, v2
s_lshl_b32 s15, s1, 8
s_mul_i32 s0, s15, s10
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s0, s2, s0
s_addc_u32 s52, s3, s1
s_lshl_b32 s14, s8, 8
s_mul_i32 s2, s14, s11
s_ashr_i32 s3, s2, 31
s_lshl_b64 s[2:3], s[2:3], 1
s_add_u32 s8, s4, s2
s_addc_u32 s53, s5, s3
v_mul_lo_u32 v12, v3, s10
v_mul_lo_u32 v13, v4, s10
v_mul_lo_u32 v14, v5, s10
v_mul_lo_u32 v15, v7, s10
v_mul_lo_u32 v16, v8, s10
v_mul_lo_u32 v17, v9, s10
v_mul_lo_u32 v26, v10, s10
v_mul_lo_u32 v27, v11, s10
v_mad_u64_u32 v[18:19], s[2:3], v3, s11, v[6:7]
v_mad_u64_u32 v[20:21], s[2:3], v4, s11, v[6:7]
v_mad_u64_u32 v[22:23], s[2:3], v5, s11, v[6:7]
v_mad_u64_u32 v[24:25], s[2:3], v7, s11, v[6:7]
s_lshl_b32 s1, s11, 8
s_ashr_i32 s4, s1, 1
s_and_b32 s1, s52, 0xffff
s_mov_b32 s3, 0x27000
s_mov_b32 s2, 0x7ffffffe
s_mul_i32 s48, s17, 0x420
s_add_i32 s5, s48, 0
v_add_lshl_u32 v4, v12, v6, 1
s_mov_b32 m0, s5
s_nop 0
buffer_load_dwordx4 v4, s[0:3], 0, offen, lds
s_add_i32 s49, s48, 0x1080
s_add_i32 s20, s5, 0x1080
v_add_lshl_u32 v5, v13, v6, 1
s_mov_b32 m0, s20
s_nop 0
buffer_load_dwordx4 v5, s[0:3], 0, offen, lds
s_add_i32 s50, s48, 0x2100
s_add_i32 s21, s5, 0x2100
v_add_lshl_u32 v7, v14, v6, 1
s_mov_b32 m0, s21
s_nop 0
buffer_load_dwordx4 v7, s[0:3], 0, offen, lds
s_add_i32 s51, s48, 0x3180
s_add_i32 s22, s5, 0x3180
v_add_lshl_u32 v8, v15, v6, 1
s_mov_b32 m0, s22
s_nop 0
buffer_load_dwordx4 v8, s[0:3], 0, offen, lds
s_add_i32 s23, s5, 0x4200
v_add_lshl_u32 v9, v16, v6, 1
s_mov_b32 m0, s23
s_nop 0
buffer_load_dwordx4 v9, s[0:3], 0, offen, lds
s_add_i32 s24, s5, 0x5280
v_add_lshl_u32 v10, v17, v6, 1
s_mov_b32 m0, s24
s_nop 0
buffer_load_dwordx4 v10, s[0:3], 0, offen, lds
s_add_i32 s25, s5, 0x6300
v_add_lshl_u32 v11, v26, v6, 1
s_mov_b32 m0, s25
s_nop 0
buffer_load_dwordx4 v11, s[0:3], 0, offen, lds
s_add_i32 s26, s5, 0x7380
v_add_lshl_u32 v12, v27, v6, 1
s_mov_b32 m0, s26
s_nop 0
buffer_load_dwordx4 v12, s[0:3], 0, offen, lds
s_and_b32 s9, s53, 0xffff
s_mov_b32 s10, s2
s_mov_b32 s11, s3
s_add_i32 s1, 0, 0x107e0
s_add_i32 s27, s1, s48
v_lshlrev_b32_e32 v13, 1, v18
s_mov_b32 m0, s27
s_nop 0
buffer_load_dwordx4 v13, s[8:11], 0, offen, lds
s_add_i32 s28, s1, s49
v_lshlrev_b32_e32 v14, 1, v20
s_mov_b32 m0, s28
s_nop 0
buffer_load_dwordx4 v14, s[8:11], 0, offen, lds
s_add_i32 s29, s1, s50
v_lshlrev_b32_e32 v15, 1, v22
s_mov_b32 m0, s29
s_nop 0
buffer_load_dwordx4 v15, s[8:11], 0, offen, lds
s_add_i32 s30, s1, s51
v_lshlrev_b32_e32 v16, 1, v24
s_mov_b32 m0, s30
s_nop 0
buffer_load_dwordx4 v16, s[8:11], 0, offen, lds
s_add_i32 s35, 0, 0x18bc0
s_add_i32 s31, s35, s48
v_add_lshl_u32 v17, v18, s4, 1
s_mov_b32 m0, s31
s_nop 0
buffer_load_dwordx4 v17, s[8:11], 0, offen, lds
s_add_i32 s33, s35, s49
v_add_lshl_u32 v18, v20, s4, 1
s_mov_b32 m0, s33
s_nop 0
buffer_load_dwordx4 v18, s[8:11], 0, offen, lds
s_add_i32 s34, s35, s50
v_add_lshl_u32 v19, v22, s4, 1
s_mov_b32 m0, s34
s_nop 0
buffer_load_dwordx4 v19, s[8:11], 0, offen, lds
s_add_i32 s35, s35, s51
v_add_lshl_u32 v20, v24, s4, 1
s_mov_b32 m0, s35
s_nop 0
buffer_load_dwordx4 v20, s[8:11], 0, offen, lds
s_add_u32 s44, s0, 0x80
s_addc_u32 s4, s52, 0
s_add_u32 s16, s8, 0x80
s_addc_u32 s9, s53, 0
s_waitcnt lgkmcnt(0)
s_barrier
s_and_b32 s45, s4, 0xffff
s_mov_b32 s46, s2
s_mov_b32 s47, s3
s_add_i32 s36, s31, 0xfffef840
s_mov_b32 m0, s36
s_nop 0
buffer_load_dwordx4 v4, s[44:47], 0, offen, lds
s_add_i32 s37, s31, 0xffff08c0
s_mov_b32 m0, s37
s_nop 0
buffer_load_dwordx4 v5, s[44:47], 0, offen, lds
s_add_i32 s38, s31, 0xffff1940
s_mov_b32 m0, s38
s_nop 0
buffer_load_dwordx4 v7, s[44:47], 0, offen, lds
s_add_i32 s39, s31, 0xffff29c0
s_mov_b32 m0, s39
s_nop 0
buffer_load_dwordx4 v8, s[44:47], 0, offen, lds
s_add_i32 s40, s31, 0xffff3a40
s_mov_b32 m0, s40
s_nop 0
buffer_load_dwordx4 v9, s[44:47], 0, offen, lds
s_add_i32 s41, s31, 0xffff4ac0
s_mov_b32 m0, s41
s_nop 0
buffer_load_dwordx4 v10, s[44:47], 0, offen, lds
s_add_i32 s42, s31, 0xffff5b40
s_mov_b32 m0, s42
s_nop 0
buffer_load_dwordx4 v11, s[44:47], 0, offen, lds
s_add_i32 s43, s31, 0xffff6bc0
s_mov_b32 m0, s43
s_nop 0
buffer_load_dwordx4 v12, s[44:47], 0, offen, lds
s_and_b32 s17, s9, 0xffff
s_mov_b32 s18, s2
s_mov_b32 s19, s3
s_add_i32 s47, 0, 0x149e0
s_add_i32 s44, s47, s48
s_mov_b32 m0, s44
s_nop 0
buffer_load_dwordx4 v13, s[16:19], 0, offen, lds
s_add_i32 s45, s47, s49
s_mov_b32 m0, s45
s_nop 0
buffer_load_dwordx4 v14, s[16:19], 0, offen, lds
s_add_i32 s46, s47, s50
s_mov_b32 m0, s46
s_nop 0
buffer_load_dwordx4 v15, s[16:19], 0, offen, lds
s_add_i32 s47, s47, s51
s_mov_b32 m0, s47
s_nop 0
buffer_load_dwordx4 v16, s[16:19], 0, offen, lds
s_add_i32 s4, 0, 0x1cdc0
s_add_i32 s48, s4, s48
s_mov_b32 m0, s48
s_nop 0
buffer_load_dwordx4 v17, s[16:19], 0, offen, lds
s_add_i32 s49, s4, s49
s_mov_b32 m0, s49
s_nop 0
buffer_load_dwordx4 v18, s[16:19], 0, offen, lds
s_add_i32 s50, s4, s50
s_mov_b32 m0, s50
s_nop 0
buffer_load_dwordx4 v19, s[16:19], 0, offen, lds
s_add_i32 s51, s4, s51
s_mov_b32 m0, s51
s_nop 0
buffer_load_dwordx4 v20, s[16:19], 0, offen, lds
s_waitcnt vmcnt(20), lgkmcnt(0)
s_barrier
v_and_b32_e32 v21, 15, v0
v_lshlrev_b32_e32 v6, 10, v21
s_movk_i32 s4, 0xb0
v_and_or_b32 v3, v1, s4, v6
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
s_and_b32 s4, s13, 64
v_and_or_b32 v6, v0, 48, v6
v_add_u32_e32 v6, v6, v22
v_lshl_add_u32 v6, s4, 1, v6
v_add_u32_e32 v22, s1, v6
ds_read_b128 v[90:93], v22
ds_read_b128 v[98:101], v22, offset:64
ds_read_b128 v[94:97], v22, offset:256
ds_read_b128 v[102:105], v22, offset:320
ds_read_b128 v[106:109], v22, offset:512
ds_read_b128 v[110:113], v22, offset:576
ds_read_b128 v[114:117], v22, offset:768
ds_read_b128 v[118:121], v22, offset:832
s_add_u32 s16, s8, 0x180
s_addc_u32 s17, s53, 0
s_add_u32 s18, s0, 0x180
s_addc_u32 s19, s52, 0
v_mov_b32_e32 v122, 0
s_mov_b32 s52, -2
v_add_u32_e32 v6, 0, v6
v_add_u32_e32 v22, 0x18bc0, v6
v_add_u32_e32 v23, 0x149e0, v6
v_add_u32_e32 v24, 0x1cdc0, v6
v_add_u32_e32 v25, 0x107e0, v6
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
s_mov_b32 s10, s2
s_mov_b32 s11, s3
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
.LBB0_1:
v_mfma_f32_16x16x32_f16 a[252:255], v[90:93], v[82:85], a[252:255]
s_add_u32 s8, s16, 0xffffff80
s_addc_u32 s9, s17, -1
s_add_u32 s0, s18, 0xffffff80
s_addc_u32 s1, s19, -1
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
s_and_b32 s1, s1, 0xffff
s_mov_b32 m0, s5
v_mfma_f32_16x16x32_f16 a[152:155], v[114:117], v[74:77], a[152:155]
buffer_load_dwordx4 v4, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[152:155], v[118:121], v[78:81], a[152:155]
v_mfma_f32_16x16x32_f16 a[156:159], v[90:93], v[66:69], a[156:159]
v_mfma_f32_16x16x32_f16 a[156:159], v[98:101], v[70:73], a[156:159]
s_mov_b32 m0, s20
v_mfma_f32_16x16x32_f16 a[160:163], v[94:97], v[66:69], a[160:163]
buffer_load_dwordx4 v5, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[160:163], v[102:105], v[70:73], a[160:163]
v_mfma_f32_16x16x32_f16 a[164:167], v[106:109], v[66:69], a[164:167]
v_mfma_f32_16x16x32_f16 a[164:167], v[110:113], v[70:73], a[164:167]
s_mov_b32 m0, s21
v_mfma_f32_16x16x32_f16 a[168:171], v[114:117], v[66:69], a[168:171]
buffer_load_dwordx4 v7, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[168:171], v[118:121], v[70:73], a[168:171]
v_mfma_f32_16x16x32_f16 a[172:175], v[90:93], v[58:61], a[172:175]
v_mfma_f32_16x16x32_f16 a[172:175], v[98:101], v[62:65], a[172:175]
s_mov_b32 m0, s22
v_mfma_f32_16x16x32_f16 a[176:179], v[94:97], v[58:61], a[176:179]
buffer_load_dwordx4 v8, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[176:179], v[102:105], v[62:65], a[176:179]
v_mfma_f32_16x16x32_f16 a[180:183], v[106:109], v[58:61], a[180:183]
v_mfma_f32_16x16x32_f16 a[180:183], v[110:113], v[62:65], a[180:183]
s_mov_b32 m0, s23
v_mfma_f32_16x16x32_f16 a[184:187], v[114:117], v[58:61], a[184:187]
buffer_load_dwordx4 v9, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[184:187], v[118:121], v[62:65], a[184:187]
v_mfma_f32_16x16x32_f16 a[188:191], v[90:93], v[50:53], a[188:191]
v_mfma_f32_16x16x32_f16 a[188:191], v[98:101], v[54:57], a[188:191]
s_mov_b32 m0, s24
v_mfma_f32_16x16x32_f16 a[192:195], v[94:97], v[50:53], a[192:195]
buffer_load_dwordx4 v10, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[192:195], v[102:105], v[54:57], a[192:195]
v_mfma_f32_16x16x32_f16 a[196:199], v[106:109], v[50:53], a[196:199]
v_mfma_f32_16x16x32_f16 a[196:199], v[110:113], v[54:57], a[196:199]
s_mov_b32 m0, s25
v_mfma_f32_16x16x32_f16 a[200:203], v[114:117], v[50:53], a[200:203]
buffer_load_dwordx4 v11, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[200:203], v[118:121], v[54:57], a[200:203]
v_mfma_f32_16x16x32_f16 a[204:207], v[90:93], v[42:45], a[204:207]
v_mfma_f32_16x16x32_f16 a[204:207], v[98:101], v[46:49], a[204:207]
s_mov_b32 m0, s26
v_mfma_f32_16x16x32_f16 a[208:211], v[94:97], v[42:45], a[208:211]
buffer_load_dwordx4 v12, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[208:211], v[102:105], v[46:49], a[208:211]
v_mfma_f32_16x16x32_f16 a[212:215], v[106:109], v[42:45], a[212:215]
v_mfma_f32_16x16x32_f16 a[212:215], v[110:113], v[46:49], a[212:215]
s_and_b32 s9, s9, 0xffff
s_mov_b32 m0, s27
v_mfma_f32_16x16x32_f16 a[216:219], v[114:117], v[42:45], a[216:219]
buffer_load_dwordx4 v13, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[216:219], v[118:121], v[46:49], a[216:219]
v_mfma_f32_16x16x32_f16 a[220:223], v[90:93], v[34:37], a[220:223]
v_mfma_f32_16x16x32_f16 a[220:223], v[98:101], v[38:41], a[220:223]
s_mov_b32 m0, s28
v_mfma_f32_16x16x32_f16 a[224:227], v[94:97], v[34:37], a[224:227]
buffer_load_dwordx4 v14, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[224:227], v[102:105], v[38:41], a[224:227]
v_mfma_f32_16x16x32_f16 a[228:231], v[106:109], v[34:37], a[228:231]
v_mfma_f32_16x16x32_f16 a[228:231], v[110:113], v[38:41], a[228:231]
s_mov_b32 m0, s29
v_mfma_f32_16x16x32_f16 a[232:235], v[114:117], v[34:37], a[232:235]
buffer_load_dwordx4 v15, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[232:235], v[118:121], v[38:41], a[232:235]
v_mfma_f32_16x16x32_f16 a[236:239], v[90:93], v[26:29], a[236:239]
v_mfma_f32_16x16x32_f16 a[236:239], v[98:101], v[30:33], a[236:239]
s_mov_b32 m0, s30
v_mfma_f32_16x16x32_f16 a[240:243], v[94:97], v[26:29], a[240:243]
buffer_load_dwordx4 v16, s[8:11], 0, offen, lds
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
s_mov_b32 m0, s31
v_mfma_f32_16x16x32_f16 a[88:91], v[146:149], v[42:45], a[88:91]
buffer_load_dwordx4 v17, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[88:91], v[150:153], v[46:49], a[88:91]
v_mfma_f32_16x16x32_f16 a[92:95], v[122:125], v[34:37], a[92:95]
v_mfma_f32_16x16x32_f16 a[92:95], v[126:129], v[38:41], a[92:95]
s_mov_b32 m0, s33
v_mfma_f32_16x16x32_f16 a[96:99], v[130:133], v[34:37], a[96:99]
buffer_load_dwordx4 v18, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[96:99], v[134:137], v[38:41], a[96:99]
v_mfma_f32_16x16x32_f16 a[100:103], v[138:141], v[34:37], a[100:103]
v_mfma_f32_16x16x32_f16 a[100:103], v[142:145], v[38:41], a[100:103]
s_mov_b32 m0, s34
v_mfma_f32_16x16x32_f16 a[104:107], v[146:149], v[34:37], a[104:107]
buffer_load_dwordx4 v19, s[8:11], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[104:107], v[150:153], v[38:41], a[104:107]
v_mfma_f32_16x16x32_f16 a[108:111], v[122:125], v[26:29], a[108:111]
v_mfma_f32_16x16x32_f16 a[108:111], v[126:129], v[30:33], a[108:111]
s_mov_b32 m0, s35
v_mfma_f32_16x16x32_f16 a[112:115], v[130:133], v[26:29], a[112:115]
buffer_load_dwordx4 v20, s[8:11], 0, offen, lds
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
s_and_b32 s1, s19, 0xffff
s_mov_b32 s0, s18
s_mov_b32 m0, s36
v_mfma_f32_16x16x32_f16 a[152:155], v[114:117], v[82:85], a[152:155]
buffer_load_dwordx4 v4, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[152:155], v[118:121], v[66:69], a[152:155]
v_mfma_f32_16x16x32_f16 a[156:159], v[58:61], v[70:73], a[156:159]
v_mfma_f32_16x16x32_f16 a[156:159], v[62:65], v[90:93], a[156:159]
s_mov_b32 m0, s37
v_mfma_f32_16x16x32_f16 a[160:163], v[86:89], v[70:73], a[160:163]
buffer_load_dwordx4 v5, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[160:163], v[50:53], v[90:93], a[160:163]
v_mfma_f32_16x16x32_f16 a[164:167], v[54:57], v[70:73], a[164:167]
v_mfma_f32_16x16x32_f16 a[164:167], v[110:113], v[90:93], a[164:167]
s_mov_b32 m0, s38
v_mfma_f32_16x16x32_f16 a[168:171], v[114:117], v[70:73], a[168:171]
buffer_load_dwordx4 v7, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[168:171], v[118:121], v[90:93], a[168:171]
v_mfma_f32_16x16x32_f16 a[172:175], v[58:61], v[94:97], a[172:175]
v_mfma_f32_16x16x32_f16 a[172:175], v[62:65], v[98:101], a[172:175]
s_mov_b32 m0, s39
v_mfma_f32_16x16x32_f16 a[176:179], v[86:89], v[94:97], a[176:179]
buffer_load_dwordx4 v8, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[176:179], v[50:53], v[98:101], a[176:179]
v_mfma_f32_16x16x32_f16 a[180:183], v[54:57], v[94:97], a[180:183]
v_mfma_f32_16x16x32_f16 a[180:183], v[110:113], v[98:101], a[180:183]
s_mov_b32 m0, s40
v_mfma_f32_16x16x32_f16 a[184:187], v[114:117], v[94:97], a[184:187]
buffer_load_dwordx4 v9, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[184:187], v[118:121], v[98:101], a[184:187]
v_mfma_f32_16x16x32_f16 a[188:191], v[58:61], v[102:105], a[188:191]
v_mfma_f32_16x16x32_f16 a[188:191], v[62:65], v[106:109], a[188:191]
s_mov_b32 m0, s41
v_mfma_f32_16x16x32_f16 a[192:195], v[86:89], v[102:105], a[192:195]
buffer_load_dwordx4 v10, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[192:195], v[50:53], v[106:109], a[192:195]
v_mfma_f32_16x16x32_f16 a[196:199], v[54:57], v[102:105], a[196:199]
v_mfma_f32_16x16x32_f16 a[196:199], v[110:113], v[106:109], a[196:199]
s_mov_b32 m0, s42
v_mfma_f32_16x16x32_f16 a[200:203], v[114:117], v[102:105], a[200:203]
buffer_load_dwordx4 v11, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[200:203], v[118:121], v[106:109], a[200:203]
v_mfma_f32_16x16x32_f16 a[204:207], v[58:61], v[154:157], a[204:207]
v_mfma_f32_16x16x32_f16 a[204:207], v[62:65], v[158:161], a[204:207]
s_mov_b32 m0, s43
v_mfma_f32_16x16x32_f16 a[208:211], v[86:89], v[154:157], a[208:211]
buffer_load_dwordx4 v12, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[208:211], v[50:53], v[158:161], a[208:211]
v_mfma_f32_16x16x32_f16 a[212:215], v[54:57], v[154:157], a[212:215]
v_mfma_f32_16x16x32_f16 a[212:215], v[110:113], v[158:161], a[212:215]
s_and_b32 s1, s17, 0xffff
s_mov_b32 s0, s16
s_mov_b32 m0, s44
v_mfma_f32_16x16x32_f16 a[216:219], v[114:117], v[154:157], a[216:219]
buffer_load_dwordx4 v13, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[216:219], v[118:121], v[158:161], a[216:219]
v_mfma_f32_16x16x32_f16 a[220:223], v[58:61], v[162:165], a[220:223]
v_mfma_f32_16x16x32_f16 a[220:223], v[62:65], v[166:169], a[220:223]
s_mov_b32 m0, s45
v_mfma_f32_16x16x32_f16 a[224:227], v[86:89], v[162:165], a[224:227]
buffer_load_dwordx4 v14, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[224:227], v[50:53], v[166:169], a[224:227]
v_mfma_f32_16x16x32_f16 a[228:231], v[54:57], v[162:165], a[228:231]
v_mfma_f32_16x16x32_f16 a[228:231], v[110:113], v[166:169], a[228:231]
s_mov_b32 m0, s46
v_mfma_f32_16x16x32_f16 a[232:235], v[114:117], v[162:165], a[232:235]
buffer_load_dwordx4 v15, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[232:235], v[118:121], v[166:169], a[232:235]
v_mfma_f32_16x16x32_f16 a[236:239], v[58:61], v[170:173], a[236:239]
v_mfma_f32_16x16x32_f16 a[236:239], v[62:65], v[174:177], a[236:239]
s_mov_b32 m0, s47
v_mfma_f32_16x16x32_f16 a[240:243], v[86:89], v[170:173], a[240:243]
buffer_load_dwordx4 v16, s[0:3], 0, offen, lds
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
s_mov_b32 m0, s48
v_mfma_f32_16x16x32_f16 a[88:91], v[146:149], v[154:157], a[88:91]
buffer_load_dwordx4 v17, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[88:91], v[150:153], v[158:161], a[88:91]
v_mfma_f32_16x16x32_f16 a[92:95], v[122:125], v[162:165], a[92:95]
v_mfma_f32_16x16x32_f16 a[92:95], v[126:129], v[166:169], a[92:95]
s_mov_b32 m0, s49
v_mfma_f32_16x16x32_f16 a[96:99], v[130:133], v[162:165], a[96:99]
buffer_load_dwordx4 v18, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[96:99], v[134:137], v[166:169], a[96:99]
v_mfma_f32_16x16x32_f16 a[100:103], v[138:141], v[162:165], a[100:103]
v_mfma_f32_16x16x32_f16 a[100:103], v[142:145], v[166:169], a[100:103]
s_mov_b32 m0, s50
v_mfma_f32_16x16x32_f16 a[104:107], v[146:149], v[162:165], a[104:107]
buffer_load_dwordx4 v19, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[104:107], v[150:153], v[166:169], a[104:107]
v_mfma_f32_16x16x32_f16 a[108:111], v[122:125], v[170:173], a[108:111]
v_mfma_f32_16x16x32_f16 a[108:111], v[126:129], v[174:177], a[108:111]
s_mov_b32 m0, s51
v_mfma_f32_16x16x32_f16 a[112:115], v[130:133], v[170:173], a[112:115]
buffer_load_dwordx4 v20, s[0:3], 0, offen, lds
v_mfma_f32_16x16x32_f16 a[112:115], v[134:137], v[174:177], a[112:115]
s_add_u32 s16, s16, 0x100
s_addc_u32 s17, s17, 0
s_add_u32 s18, s18, 0x100
s_addc_u32 s19, s19, 0
s_add_i32 s52, s52, 2
v_mfma_f32_16x16x32_f16 a[116:119], v[138:141], v[170:173], a[116:119]
s_cmpk_lt_u32 s52, 0x7c
s_waitcnt vmcnt(16), lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x32_f16 a[116:119], v[142:145], v[174:177], a[116:119]
v_mfma_f32_16x16x32_f16 a[120:123], v[146:149], v[170:173], a[120:123]
v_mfma_f32_16x16x32_f16 a[120:123], v[150:153], v[174:177], a[120:123]
s_cbranch_scc1 .LBB0_1
; %bb.2:
v_lshrrev_b32_e32 v5, 4, v1
v_or_b32_e32 v7, 16, v5
v_or_b32_e32 v8, 32, v5
v_or_b32_e32 v9, 48, v5
v_or_b32_e32 v10, 64, v5
v_or_b32_e32 v11, 0x50, v5
s_movk_i32 s2, 0x60
v_or_b32_e32 v12, 0x60, v5
v_or_b32_e32 v13, 0x70, v5
v_or_b32_e32 v22, 0x80, v5
v_or_b32_e32 v23, 0x90, v5
v_or_b32_e32 v24, 0xa0, v5
v_or_b32_e32 v25, 0xb0, v5
v_or_b32_e32 v122, 0xc0, v5
v_or_b32_e32 v123, 0xd0, v5
v_or_b32_e32 v124, 0xe0, v5
v_or_b32_e32 v125, 0xf0, v5
v_lshlrev_b32_e32 v4, 3, v21
s_mul_i32 s0, s15, s12
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s3, s6, s0
s_addc_u32 s5, s7, s1
s_ashr_i32 s15, s14, 31
s_lshl_b64 s[0:1], s[14:15], 1
s_add_u32 s0, s3, s0
s_addc_u32 s1, s5, s1
v_mul_lo_u32 v21, v5, s12
v_mul_lo_u32 v20, v7, s12
v_mul_lo_u32 v19, v8, s12
v_mul_lo_u32 v18, v9, s12
v_mul_lo_u32 v17, v10, s12
v_mul_lo_u32 v16, v11, s12
v_mul_lo_u32 v15, v12, s12
v_mul_lo_u32 v14, v13, s12
v_mul_lo_u32 v13, v22, s12
v_mul_lo_u32 v12, v23, s12
v_mul_lo_u32 v11, v24, s12
v_mul_lo_u32 v10, v25, s12
v_mul_lo_u32 v9, v122, s12
v_mul_lo_u32 v8, v123, s12
v_mul_lo_u32 v7, v124, s12
v_mul_lo_u32 v5, v125, s12
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_f16 a[252:255], v[90:93], v[82:85], a[252:255]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_f16 a[252:255], v[98:101], v[86:89], a[252:255]
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_f16 a[128:131], v[94:97], v[82:85], a[128:131]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_f16 a[128:131], v[102:105], v[86:89], a[128:131]
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_f16 a[132:135], v[106:109], v[82:85], a[132:135]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_f16 a[132:135], v[110:113], v[86:89], a[132:135]
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_f16 a[136:139], v[114:117], v[82:85], a[136:139]
s_waitcnt lgkmcnt(0)
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
s_waitcnt vmcnt(0), lgkmcnt(0)
s_barrier
v_add_u32_e32 v22, 0x18bc0, v6
ds_read_b128 v[90:93], v22
ds_read_b128 v[94:97], v22, offset:64
ds_read_b128 v[98:101], v22, offset:256
ds_read_b128 v[102:105], v22, offset:320
ds_read_b128 v[106:109], v22, offset:512
ds_read_b128 v[110:113], v22, offset:576
ds_read_b128 v[114:117], v22, offset:768
ds_read_b128 v[22:25], v22, offset:832
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_f16 a[124:127], v[90:93], v[82:85], a[124:127]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_f16 a[124:127], v[94:97], v[86:89], a[124:127]
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_f16 a[0:3], v[98:101], v[82:85], a[0:3]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_f16 a[0:3], v[102:105], v[86:89], a[0:3]
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_f16 a[4:7], v[106:109], v[82:85], a[4:7]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_f16 a[4:7], v[110:113], v[86:89], a[4:7]
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_f16 a[8:11], v[114:117], v[82:85], a[8:11]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[8:11], v[22:25], v[86:89], a[8:11]
v_mfma_f32_16x16x32_f16 a[12:15], v[90:93], v[74:77], a[12:15]
v_mfma_f32_16x16x32_f16 a[12:15], v[94:97], v[78:81], a[12:15]
v_mfma_f32_16x16x32_f16 a[16:19], v[98:101], v[74:77], a[16:19]
v_mfma_f32_16x16x32_f16 a[16:19], v[102:105], v[78:81], a[16:19]
v_mfma_f32_16x16x32_f16 a[20:23], v[106:109], v[74:77], a[20:23]
v_mfma_f32_16x16x32_f16 a[20:23], v[110:113], v[78:81], a[20:23]
v_mfma_f32_16x16x32_f16 a[24:27], v[114:117], v[74:77], a[24:27]
v_mfma_f32_16x16x32_f16 a[24:27], v[22:25], v[78:81], a[24:27]
v_mfma_f32_16x16x32_f16 a[28:31], v[90:93], v[66:69], a[28:31]
v_mfma_f32_16x16x32_f16 a[28:31], v[94:97], v[70:73], a[28:31]
v_mfma_f32_16x16x32_f16 a[32:35], v[98:101], v[66:69], a[32:35]
v_mfma_f32_16x16x32_f16 a[32:35], v[102:105], v[70:73], a[32:35]
v_mfma_f32_16x16x32_f16 a[36:39], v[106:109], v[66:69], a[36:39]
v_mfma_f32_16x16x32_f16 a[36:39], v[110:113], v[70:73], a[36:39]
v_mfma_f32_16x16x32_f16 a[40:43], v[114:117], v[66:69], a[40:43]
v_mfma_f32_16x16x32_f16 a[40:43], v[22:25], v[70:73], a[40:43]
v_mfma_f32_16x16x32_f16 a[44:47], v[90:93], v[58:61], a[44:47]
v_mfma_f32_16x16x32_f16 a[44:47], v[94:97], v[62:65], a[44:47]
v_mfma_f32_16x16x32_f16 a[48:51], v[98:101], v[58:61], a[48:51]
v_mfma_f32_16x16x32_f16 a[48:51], v[102:105], v[62:65], a[48:51]
v_mfma_f32_16x16x32_f16 a[52:55], v[106:109], v[58:61], a[52:55]
v_mfma_f32_16x16x32_f16 a[52:55], v[110:113], v[62:65], a[52:55]
v_mfma_f32_16x16x32_f16 a[56:59], v[114:117], v[58:61], a[56:59]
v_mfma_f32_16x16x32_f16 a[56:59], v[22:25], v[62:65], a[56:59]
v_mfma_f32_16x16x32_f16 a[60:63], v[90:93], v[50:53], a[60:63]
v_mfma_f32_16x16x32_f16 a[60:63], v[94:97], v[54:57], a[60:63]
v_mfma_f32_16x16x32_f16 a[64:67], v[98:101], v[50:53], a[64:67]
v_mfma_f32_16x16x32_f16 a[64:67], v[102:105], v[54:57], a[64:67]
v_mfma_f32_16x16x32_f16 a[68:71], v[106:109], v[50:53], a[68:71]
v_mfma_f32_16x16x32_f16 a[68:71], v[110:113], v[54:57], a[68:71]
v_mfma_f32_16x16x32_f16 a[72:75], v[114:117], v[50:53], a[72:75]
v_mfma_f32_16x16x32_f16 a[72:75], v[22:25], v[54:57], a[72:75]
v_mfma_f32_16x16x32_f16 a[76:79], v[90:93], v[42:45], a[76:79]
v_mfma_f32_16x16x32_f16 a[76:79], v[94:97], v[46:49], a[76:79]
v_mfma_f32_16x16x32_f16 a[80:83], v[98:101], v[42:45], a[80:83]
v_mfma_f32_16x16x32_f16 a[80:83], v[102:105], v[46:49], a[80:83]
v_mfma_f32_16x16x32_f16 a[84:87], v[106:109], v[42:45], a[84:87]
v_mfma_f32_16x16x32_f16 a[84:87], v[110:113], v[46:49], a[84:87]
v_mfma_f32_16x16x32_f16 a[88:91], v[114:117], v[42:45], a[88:91]
v_mfma_f32_16x16x32_f16 a[88:91], v[22:25], v[46:49], a[88:91]
v_mfma_f32_16x16x32_f16 a[92:95], v[90:93], v[34:37], a[92:95]
v_mfma_f32_16x16x32_f16 a[92:95], v[94:97], v[38:41], a[92:95]
v_mfma_f32_16x16x32_f16 a[96:99], v[98:101], v[34:37], a[96:99]
v_mfma_f32_16x16x32_f16 a[96:99], v[102:105], v[38:41], a[96:99]
v_mfma_f32_16x16x32_f16 a[100:103], v[106:109], v[34:37], a[100:103]
v_mfma_f32_16x16x32_f16 a[100:103], v[110:113], v[38:41], a[100:103]
v_mfma_f32_16x16x32_f16 a[104:107], v[114:117], v[34:37], a[104:107]
v_mfma_f32_16x16x32_f16 a[104:107], v[22:25], v[38:41], a[104:107]
v_mfma_f32_16x16x32_f16 a[108:111], v[90:93], v[26:29], a[108:111]
v_mfma_f32_16x16x32_f16 a[108:111], v[94:97], v[30:33], a[108:111]
v_mfma_f32_16x16x32_f16 a[112:115], v[98:101], v[26:29], a[112:115]
v_mfma_f32_16x16x32_f16 a[112:115], v[102:105], v[30:33], a[112:115]
v_mfma_f32_16x16x32_f16 a[116:119], v[106:109], v[26:29], a[116:119]
v_mfma_f32_16x16x32_f16 a[116:119], v[110:113], v[30:33], a[116:119]
v_mfma_f32_16x16x32_f16 a[120:123], v[114:117], v[26:29], a[120:123]
v_mfma_f32_16x16x32_f16 a[120:123], v[22:25], v[30:33], a[120:123]
ds_read_b128 v[78:81], v3, offset:33792
ds_read_b128 v[82:85], v3, offset:33856
ds_read_b128 v[70:73], v3, offset:34048
ds_read_b128 v[74:77], v3, offset:34112
ds_read_b128 v[62:65], v3, offset:34304
ds_read_b128 v[66:69], v3, offset:34368
ds_read_b128 v[54:57], v3, offset:34560
ds_read_b128 v[58:61], v3, offset:34624
ds_read_b128 v[46:49], v3, offset:50688
ds_read_b128 v[50:53], v3, offset:50752
ds_read_b128 v[38:41], v3, offset:50944
ds_read_b128 v[42:45], v3, offset:51008
ds_read_b128 v[30:33], v3, offset:51200
ds_read_b128 v[34:37], v3, offset:51264
ds_read_b128 v[22:25], v3, offset:51456
ds_read_b128 v[26:29], v3, offset:51520
v_add_u32_e32 v3, 0x149e0, v6
ds_read_b128 v[86:89], v3
ds_read_b128 v[90:93], v3, offset:64
ds_read_b128 v[94:97], v3, offset:256
ds_read_b128 v[98:101], v3, offset:320
ds_read_b128 v[102:105], v3, offset:512
ds_read_b128 v[106:109], v3, offset:576
ds_read_b128 v[110:113], v3, offset:768
ds_read_b128 v[114:117], v3, offset:832
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_f16 a[252:255], v[86:89], v[78:81], a[252:255]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_f16 a[252:255], v[90:93], v[82:85], a[252:255]
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_f16 a[128:131], v[94:97], v[78:81], a[128:131]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_f16 a[128:131], v[98:101], v[82:85], a[128:131]
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_f16 a[132:135], v[102:105], v[78:81], a[132:135]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_f16 a[132:135], v[106:109], v[82:85], a[132:135]
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_f16 a[136:139], v[110:113], v[78:81], a[136:139]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_f16 a[136:139], v[114:117], v[82:85], a[136:139]
v_mfma_f32_16x16x32_f16 a[140:143], v[86:89], v[70:73], a[140:143]
v_mfma_f32_16x16x32_f16 a[140:143], v[90:93], v[74:77], a[140:143]
v_mfma_f32_16x16x32_f16 a[144:147], v[94:97], v[70:73], a[144:147]
v_mfma_f32_16x16x32_f16 a[144:147], v[98:101], v[74:77], a[144:147]
v_mfma_f32_16x16x32_f16 a[148:151], v[102:105], v[70:73], a[148:151]
v_mfma_f32_16x16x32_f16 a[148:151], v[106:109], v[74:77], a[148:151]
v_mfma_f32_16x16x32_f16 a[152:155], v[110:113], v[70:73], a[152:155]
v_mfma_f32_16x16x32_f16 a[152:155], v[114:117], v[74:77], a[152:155]
v_mfma_f32_16x16x32_f16 a[156:159], v[86:89], v[62:65], a[156:159]
v_mfma_f32_16x16x32_f16 a[156:159], v[90:93], v[66:69], a[156:159]
v_mfma_f32_16x16x32_f16 a[160:163], v[94:97], v[62:65], a[160:163]
v_mfma_f32_16x16x32_f16 a[160:163], v[98:101], v[66:69], a[160:163]
v_mfma_f32_16x16x32_f16 a[164:167], v[102:105], v[62:65], a[164:167]
v_mfma_f32_16x16x32_f16 a[164:167], v[106:109], v[66:69], a[164:167]
v_mfma_f32_16x16x32_f16 a[168:171], v[110:113], v[62:65], a[168:171]
v_mfma_f32_16x16x32_f16 a[168:171], v[114:117], v[66:69], a[168:171]
v_mfma_f32_16x16x32_f16 a[172:175], v[86:89], v[54:57], a[172:175]
v_mfma_f32_16x16x32_f16 a[172:175], v[90:93], v[58:61], a[172:175]
v_mfma_f32_16x16x32_f16 a[176:179], v[94:97], v[54:57], a[176:179]
v_mfma_f32_16x16x32_f16 a[176:179], v[98:101], v[58:61], a[176:179]
v_mfma_f32_16x16x32_f16 a[180:183], v[102:105], v[54:57], a[180:183]
v_mfma_f32_16x16x32_f16 a[180:183], v[106:109], v[58:61], a[180:183]
v_mfma_f32_16x16x32_f16 a[184:187], v[110:113], v[54:57], a[184:187]
v_mfma_f32_16x16x32_f16 a[184:187], v[114:117], v[58:61], a[184:187]
v_mfma_f32_16x16x32_f16 a[188:191], v[86:89], v[46:49], a[188:191]
v_mfma_f32_16x16x32_f16 a[188:191], v[90:93], v[50:53], a[188:191]
v_mfma_f32_16x16x32_f16 a[192:195], v[94:97], v[46:49], a[192:195]
v_mfma_f32_16x16x32_f16 a[192:195], v[98:101], v[50:53], a[192:195]
v_mfma_f32_16x16x32_f16 a[196:199], v[102:105], v[46:49], a[196:199]
v_mfma_f32_16x16x32_f16 a[196:199], v[106:109], v[50:53], a[196:199]
v_mfma_f32_16x16x32_f16 a[200:203], v[110:113], v[46:49], a[200:203]
v_mfma_f32_16x16x32_f16 a[200:203], v[114:117], v[50:53], a[200:203]
v_mfma_f32_16x16x32_f16 a[204:207], v[86:89], v[38:41], a[204:207]
v_mfma_f32_16x16x32_f16 a[204:207], v[90:93], v[42:45], a[204:207]
v_mfma_f32_16x16x32_f16 a[208:211], v[94:97], v[38:41], a[208:211]
v_mfma_f32_16x16x32_f16 a[208:211], v[98:101], v[42:45], a[208:211]
v_mfma_f32_16x16x32_f16 a[212:215], v[102:105], v[38:41], a[212:215]
v_mfma_f32_16x16x32_f16 a[212:215], v[106:109], v[42:45], a[212:215]
v_mfma_f32_16x16x32_f16 a[216:219], v[110:113], v[38:41], a[216:219]
v_mfma_f32_16x16x32_f16 a[216:219], v[114:117], v[42:45], a[216:219]
v_mfma_f32_16x16x32_f16 a[220:223], v[86:89], v[30:33], a[220:223]
v_mfma_f32_16x16x32_f16 a[220:223], v[90:93], v[34:37], a[220:223]
v_mfma_f32_16x16x32_f16 a[224:227], v[94:97], v[30:33], a[224:227]
v_mfma_f32_16x16x32_f16 a[224:227], v[98:101], v[34:37], a[224:227]
v_mfma_f32_16x16x32_f16 a[228:231], v[102:105], v[30:33], a[228:231]
v_mfma_f32_16x16x32_f16 a[228:231], v[106:109], v[34:37], a[228:231]
v_mfma_f32_16x16x32_f16 a[232:235], v[110:113], v[30:33], a[232:235]
v_mfma_f32_16x16x32_f16 a[232:235], v[114:117], v[34:37], a[232:235]
v_mfma_f32_16x16x32_f16 a[236:239], v[86:89], v[22:25], a[236:239]
v_mfma_f32_16x16x32_f16 a[236:239], v[90:93], v[26:29], a[236:239]
v_mfma_f32_16x16x32_f16 a[240:243], v[94:97], v[22:25], a[240:243]
v_mfma_f32_16x16x32_f16 a[240:243], v[98:101], v[26:29], a[240:243]
v_mfma_f32_16x16x32_f16 a[244:247], v[102:105], v[22:25], a[244:247]
v_mfma_f32_16x16x32_f16 a[244:247], v[106:109], v[26:29], a[244:247]
v_mfma_f32_16x16x32_f16 a[248:251], v[110:113], v[22:25], a[248:251]
v_mfma_f32_16x16x32_f16 a[248:251], v[114:117], v[26:29], a[248:251]
v_add_u32_e32 v3, 0x1cdc0, v6
ds_read_b128 v[172:175], v3
ds_read_b128 v[176:179], v3, offset:64
ds_read_b128 v[180:183], v3, offset:256
ds_read_b128 v[184:187], v3, offset:320
ds_read_b128 v[188:191], v3, offset:512
ds_read_b128 v[192:195], v3, offset:576
ds_read_b128 v[196:199], v3, offset:768
ds_read_b128 v[200:203], v3, offset:832
v_accvgpr_read_b32 v6, a252
v_accvgpr_read_b32 v3, a253
v_cvt_pk_f16_f32 v86, v6, v3
v_accvgpr_read_b32 v6, a254
v_accvgpr_read_b32 v3, a255
v_cvt_pk_f16_f32 v87, v6, v3
v_accvgpr_read_b32 v6, a128
v_accvgpr_read_b32 v3, a129
v_cvt_pk_f16_f32 v90, v6, v3
v_accvgpr_read_b32 v6, a130
v_accvgpr_read_b32 v3, a131
v_cvt_pk_f16_f32 v91, v6, v3
v_accvgpr_read_b32 v6, a132
v_accvgpr_read_b32 v3, a133
v_cvt_pk_f16_f32 v94, v6, v3
v_accvgpr_read_b32 v6, a134
v_accvgpr_read_b32 v3, a135
v_cvt_pk_f16_f32 v95, v6, v3
v_accvgpr_read_b32 v6, a136
v_accvgpr_read_b32 v3, a137
v_cvt_pk_f16_f32 v98, v6, v3
v_accvgpr_read_b32 v6, a138
v_accvgpr_read_b32 v3, a139
v_cvt_pk_f16_f32 v99, v6, v3
v_accvgpr_read_b32 v6, a140
v_accvgpr_read_b32 v3, a141
v_cvt_pk_f16_f32 v88, v6, v3
v_accvgpr_read_b32 v6, a142
v_accvgpr_read_b32 v3, a143
v_cvt_pk_f16_f32 v89, v6, v3
v_accvgpr_read_b32 v6, a144
v_accvgpr_read_b32 v3, a145
v_cvt_pk_f16_f32 v92, v6, v3
v_accvgpr_read_b32 v6, a146
v_accvgpr_read_b32 v3, a147
v_cvt_pk_f16_f32 v93, v6, v3
v_accvgpr_read_b32 v6, a148
v_accvgpr_read_b32 v3, a149
v_cvt_pk_f16_f32 v96, v6, v3
v_accvgpr_read_b32 v6, a150
v_accvgpr_read_b32 v3, a151
v_cvt_pk_f16_f32 v97, v6, v3
v_accvgpr_read_b32 v6, a152
v_accvgpr_read_b32 v3, a153
v_cvt_pk_f16_f32 v100, v6, v3
v_accvgpr_read_b32 v6, a154
v_accvgpr_read_b32 v3, a155
v_cvt_pk_f16_f32 v101, v6, v3
v_accvgpr_read_b32 v6, a156
v_accvgpr_read_b32 v3, a157
v_cvt_pk_f16_f32 v102, v6, v3
v_accvgpr_read_b32 v6, a158
v_accvgpr_read_b32 v3, a159
v_cvt_pk_f16_f32 v103, v6, v3
v_accvgpr_read_b32 v6, a160
v_accvgpr_read_b32 v3, a161
v_cvt_pk_f16_f32 v106, v6, v3
v_accvgpr_read_b32 v6, a162
v_accvgpr_read_b32 v3, a163
v_cvt_pk_f16_f32 v107, v6, v3
v_accvgpr_read_b32 v6, a164
v_accvgpr_read_b32 v3, a165
v_cvt_pk_f16_f32 v110, v6, v3
v_accvgpr_read_b32 v6, a166
v_accvgpr_read_b32 v3, a167
v_cvt_pk_f16_f32 v111, v6, v3
v_accvgpr_read_b32 v6, a168
v_accvgpr_read_b32 v3, a169
v_cvt_pk_f16_f32 v114, v6, v3
v_accvgpr_read_b32 v6, a170
v_accvgpr_read_b32 v3, a171
v_cvt_pk_f16_f32 v115, v6, v3
v_accvgpr_read_b32 v6, a172
v_accvgpr_read_b32 v3, a173
v_cvt_pk_f16_f32 v104, v6, v3
v_accvgpr_read_b32 v6, a174
v_accvgpr_read_b32 v3, a175
v_cvt_pk_f16_f32 v105, v6, v3
v_accvgpr_read_b32 v6, a176
v_accvgpr_read_b32 v3, a177
v_cvt_pk_f16_f32 v108, v6, v3
v_accvgpr_read_b32 v6, a178
v_accvgpr_read_b32 v3, a179
v_cvt_pk_f16_f32 v109, v6, v3
v_accvgpr_read_b32 v6, a180
v_accvgpr_read_b32 v3, a181
v_cvt_pk_f16_f32 v112, v6, v3
v_accvgpr_read_b32 v6, a182
v_accvgpr_read_b32 v3, a183
v_cvt_pk_f16_f32 v113, v6, v3
v_accvgpr_read_b32 v6, a184
v_accvgpr_read_b32 v3, a185
v_cvt_pk_f16_f32 v116, v6, v3
v_accvgpr_read_b32 v6, a186
v_accvgpr_read_b32 v3, a187
v_cvt_pk_f16_f32 v117, v6, v3
v_accvgpr_read_b32 v6, a188
v_accvgpr_read_b32 v3, a189
v_cvt_pk_f16_f32 v118, v6, v3
v_accvgpr_read_b32 v6, a190
v_accvgpr_read_b32 v3, a191
v_cvt_pk_f16_f32 v119, v6, v3
v_accvgpr_read_b32 v6, a192
v_accvgpr_read_b32 v3, a193
v_cvt_pk_f16_f32 v122, v6, v3
v_accvgpr_read_b32 v6, a194
v_accvgpr_read_b32 v3, a195
v_cvt_pk_f16_f32 v123, v6, v3
v_accvgpr_read_b32 v6, a196
v_accvgpr_read_b32 v3, a197
v_cvt_pk_f16_f32 v126, v6, v3
v_accvgpr_read_b32 v6, a198
v_accvgpr_read_b32 v3, a199
v_cvt_pk_f16_f32 v127, v6, v3
v_accvgpr_read_b32 v6, a200
v_accvgpr_read_b32 v3, a201
v_cvt_pk_f16_f32 v130, v6, v3
v_accvgpr_read_b32 v6, a202
v_accvgpr_read_b32 v3, a203
v_cvt_pk_f16_f32 v131, v6, v3
v_accvgpr_read_b32 v6, a204
v_accvgpr_read_b32 v3, a205
v_cvt_pk_f16_f32 v120, v6, v3
v_accvgpr_read_b32 v6, a206
v_accvgpr_read_b32 v3, a207
v_cvt_pk_f16_f32 v121, v6, v3
v_accvgpr_read_b32 v6, a208
v_accvgpr_read_b32 v3, a209
v_cvt_pk_f16_f32 v124, v6, v3
v_accvgpr_read_b32 v6, a210
v_accvgpr_read_b32 v3, a211
v_cvt_pk_f16_f32 v125, v6, v3
v_accvgpr_read_b32 v6, a212
v_accvgpr_read_b32 v3, a213
v_cvt_pk_f16_f32 v128, v6, v3
v_accvgpr_read_b32 v6, a214
v_accvgpr_read_b32 v3, a215
v_cvt_pk_f16_f32 v129, v6, v3
v_accvgpr_read_b32 v6, a216
v_accvgpr_read_b32 v3, a217
v_cvt_pk_f16_f32 v132, v6, v3
v_accvgpr_read_b32 v6, a218
v_accvgpr_read_b32 v3, a219
v_cvt_pk_f16_f32 v133, v6, v3
v_accvgpr_read_b32 v6, a220
v_accvgpr_read_b32 v3, a221
v_cvt_pk_f16_f32 v134, v6, v3
v_accvgpr_read_b32 v6, a222
v_accvgpr_read_b32 v3, a223
v_cvt_pk_f16_f32 v135, v6, v3
v_accvgpr_read_b32 v6, a224
v_accvgpr_read_b32 v3, a225
v_cvt_pk_f16_f32 v138, v6, v3
v_accvgpr_read_b32 v6, a226
v_accvgpr_read_b32 v3, a227
v_cvt_pk_f16_f32 v139, v6, v3
v_accvgpr_read_b32 v6, a228
v_accvgpr_read_b32 v3, a229
v_cvt_pk_f16_f32 v142, v6, v3
v_accvgpr_read_b32 v6, a230
v_accvgpr_read_b32 v3, a231
v_cvt_pk_f16_f32 v143, v6, v3
v_accvgpr_read_b32 v6, a232
v_accvgpr_read_b32 v3, a233
v_cvt_pk_f16_f32 v146, v6, v3
v_accvgpr_read_b32 v6, a234
v_accvgpr_read_b32 v3, a235
v_cvt_pk_f16_f32 v147, v6, v3
v_accvgpr_read_b32 v6, a236
v_accvgpr_read_b32 v3, a237
v_cvt_pk_f16_f32 v136, v6, v3
v_accvgpr_read_b32 v6, a238
v_accvgpr_read_b32 v3, a239
v_cvt_pk_f16_f32 v137, v6, v3
v_accvgpr_read_b32 v6, a240
v_accvgpr_read_b32 v3, a241
v_cvt_pk_f16_f32 v140, v6, v3
v_accvgpr_read_b32 v6, a242
v_accvgpr_read_b32 v3, a243
v_cvt_pk_f16_f32 v141, v6, v3
v_accvgpr_read_b32 v6, a244
v_accvgpr_read_b32 v3, a245
v_cvt_pk_f16_f32 v144, v6, v3
v_accvgpr_read_b32 v6, a246
v_accvgpr_read_b32 v3, a247
v_cvt_pk_f16_f32 v145, v6, v3
v_accvgpr_read_b32 v6, a248
v_accvgpr_read_b32 v3, a249
v_cvt_pk_f16_f32 v148, v6, v3
v_accvgpr_read_b32 v6, a250
v_accvgpr_read_b32 v3, a251
v_cvt_pk_f16_f32 v149, v6, v3
s_waitcnt lgkmcnt(0)
s_barrier
v_lshlrev_b32_e32 v3, 8, v0
v_and_b32_e32 v150, 0x70, v2
v_and_b32_e32 v151, 1, v0
v_lshlrev_b32_e32 v6, 12, v151
v_and_b32_e32 v152, 16, v0
v_lshlrev_b32_e32 v0, 4, v152
s_lshr_b32 s3, s4, 2
s_and_b32 s4, s13, 0x80
s_movk_i32 s5, 0x2e00
v_and_or_b32 v3, v3, s5, v6
v_or3_b32 v6, s4, v0, v3
v_mov_b32_e32 v0, 0x70
v_bitop3_b32 v153, s3, v2, v0, bitop3:0x78
v_or_b32_e32 v3, v6, v153
v_add_u32_e32 v0, 0, v3
ds_write_b128 v0, v[86:89]
v_xad_u32 v2, v3, 32, 0
ds_write_b128 v2, v[90:93]
v_xad_u32 v3, v3, 64, 0
ds_write_b128 v3, v[94:97]
v_bitop3_b32 v6, v6, s2, v153, bitop3:0x36
v_add_u32_e32 v6, 0, v6
ds_write_b128 v6, v[98:101]
s_waitcnt lgkmcnt(0)
s_barrier
v_and_b32_e32 v1, 0xe0, v1
v_lshlrev_b32_e32 v86, 4, v1
v_lshrrev_b32_e32 v1, 1, v1
v_lshlrev_b32_e32 v87, 8, v152
v_bitop3_b32 v1, v86, v1, v150, bitop3:0x36
v_lshl_add_u32 v86, v151, 13, 0
v_add3_u32 v1, v86, v87, v1
ds_read_b128 v[88:91], v1
ds_read_b128 v[94:97], v1, offset:256
ds_read_b128 v[98:101], v1, offset:128
ds_read_b128 v[150:153], v1, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[102:105]
ds_write_b128 v2, v[106:109]
ds_write_b128 v3, v[110:113]
ds_write_b128 v6, v[114:117]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[104:107], v1
ds_read_b128 v[110:113], v1, offset:256
ds_read_b128 v[114:117], v1, offset:128
ds_read_b128 v[154:157], v1, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[118:121]
ds_write_b128 v2, v[122:125]
ds_write_b128 v3, v[126:129]
ds_write_b128 v6, v[130:133]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[120:123], v1
ds_read_b128 v[126:129], v1, offset:256
ds_read_b128 v[130:133], v1, offset:128
ds_read_b128 v[158:161], v1, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[134:137]
ds_write_b128 v2, v[138:141]
ds_write_b128 v3, v[142:145]
ds_write_b128 v6, v[146:149]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[136:139], v1
ds_read_b128 v[142:145], v1, offset:256
ds_read_b128 v[162:165], v1, offset:128
ds_read_b128 v[168:171], v1, offset:384
s_and_b32 s1, s1, 0xffff
s_mov_b32 s3, 0x27000
s_mov_b32 s2, 0x7ffffffe
v_mov_b32_e32 v92, v88
v_mov_b32_e32 v93, v89
v_add_lshl_u32 v86, v21, v4, 1
buffer_store_dwordx4 v[92:95], v86, s[0:3], 0, offen
v_mov_b32_e32 v148, v98
v_mov_b32_e32 v149, v99
v_add_lshl_u32 v87, v20, v4, 1
buffer_store_dwordx4 v[148:151], v87, s[0:3], 0, offen
v_mov_b32_e32 v92, v96
v_mov_b32_e32 v93, v97
v_add_lshl_u32 v88, v19, v4, 1
buffer_store_dwordx4 v[90:93], v88, s[0:3], 0, offen
v_mov_b32_e32 v102, v152
v_mov_b32_e32 v103, v153
v_add_lshl_u32 v89, v18, v4, 1
buffer_store_dwordx4 v[100:103], v89, s[0:3], 0, offen
v_mov_b32_e32 v108, v104
v_mov_b32_e32 v109, v105
v_add_lshl_u32 v90, v17, v4, 1
buffer_store_dwordx4 v[108:111], v90, s[0:3], 0, offen
v_mov_b32_e32 v152, v114
v_mov_b32_e32 v153, v115
v_add_lshl_u32 v91, v16, v4, 1
buffer_store_dwordx4 v[152:155], v91, s[0:3], 0, offen
v_mov_b32_e32 v108, v112
v_mov_b32_e32 v109, v113
v_add_lshl_u32 v92, v15, v4, 1
buffer_store_dwordx4 v[106:109], v92, s[0:3], 0, offen
v_mov_b32_e32 v118, v156
v_mov_b32_e32 v119, v157
v_add_lshl_u32 v93, v14, v4, 1
buffer_store_dwordx4 v[116:119], v93, s[0:3], 0, offen
v_mov_b32_e32 v124, v120
v_mov_b32_e32 v125, v121
v_add_lshl_u32 v94, v13, v4, 1
buffer_store_dwordx4 v[124:127], v94, s[0:3], 0, offen
v_mov_b32_e32 v156, v130
v_mov_b32_e32 v157, v131
v_add_lshl_u32 v95, v12, v4, 1
buffer_store_dwordx4 v[156:159], v95, s[0:3], 0, offen
v_mov_b32_e32 v124, v128
v_mov_b32_e32 v125, v129
v_add_lshl_u32 v96, v11, v4, 1
buffer_store_dwordx4 v[122:125], v96, s[0:3], 0, offen
v_mov_b32_e32 v134, v160
v_mov_b32_e32 v135, v161
v_add_lshl_u32 v97, v10, v4, 1
buffer_store_dwordx4 v[132:135], v97, s[0:3], 0, offen
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v140, v136
v_mov_b32_e32 v141, v137
v_add_lshl_u32 v98, v9, v4, 1
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[140:143], v98, s[0:3], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v166, v162
v_mov_b32_e32 v167, v163
v_add_lshl_u32 v99, v8, v4, 1
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[166:169], v99, s[0:3], 0, offen
v_mov_b32_e32 v140, v144
v_mov_b32_e32 v141, v145
v_add_lshl_u32 v7, v7, v4, 1
buffer_store_dwordx4 v[138:141], v7, s[0:3], 0, offen
v_mov_b32_e32 v166, v170
v_mov_b32_e32 v167, v171
v_add_lshl_u32 v100, v5, v4, 1
buffer_store_dwordx4 v[164:167], v100, s[0:3], 0, offen
v_mfma_f32_16x16x32_f16 a[124:127], v[172:175], v[78:81], a[124:127]
v_mfma_f32_16x16x32_f16 a[124:127], v[176:179], v[82:85], a[124:127]
v_mfma_f32_16x16x32_f16 a[0:3], v[180:183], v[78:81], a[0:3]
v_mfma_f32_16x16x32_f16 a[0:3], v[184:187], v[82:85], a[0:3]
v_mfma_f32_16x16x32_f16 a[4:7], v[188:191], v[78:81], a[4:7]
v_mfma_f32_16x16x32_f16 a[4:7], v[192:195], v[82:85], a[4:7]
v_mfma_f32_16x16x32_f16 a[8:11], v[196:199], v[78:81], a[8:11]
v_mfma_f32_16x16x32_f16 a[8:11], v[200:203], v[82:85], a[8:11]
v_mfma_f32_16x16x32_f16 a[12:15], v[172:175], v[70:73], a[12:15]
v_mfma_f32_16x16x32_f16 a[12:15], v[176:179], v[74:77], a[12:15]
v_mfma_f32_16x16x32_f16 a[16:19], v[180:183], v[70:73], a[16:19]
v_mfma_f32_16x16x32_f16 a[16:19], v[184:187], v[74:77], a[16:19]
v_mfma_f32_16x16x32_f16 a[20:23], v[188:191], v[70:73], a[20:23]
v_mfma_f32_16x16x32_f16 a[20:23], v[192:195], v[74:77], a[20:23]
v_mfma_f32_16x16x32_f16 a[24:27], v[196:199], v[70:73], a[24:27]
v_mfma_f32_16x16x32_f16 a[24:27], v[200:203], v[74:77], a[24:27]
v_mfma_f32_16x16x32_f16 a[28:31], v[172:175], v[62:65], a[28:31]
v_mfma_f32_16x16x32_f16 a[28:31], v[176:179], v[66:69], a[28:31]
v_mfma_f32_16x16x32_f16 a[32:35], v[180:183], v[62:65], a[32:35]
v_mfma_f32_16x16x32_f16 a[32:35], v[184:187], v[66:69], a[32:35]
v_mfma_f32_16x16x32_f16 a[36:39], v[188:191], v[62:65], a[36:39]
v_mfma_f32_16x16x32_f16 a[36:39], v[192:195], v[66:69], a[36:39]
v_mfma_f32_16x16x32_f16 a[40:43], v[196:199], v[62:65], a[40:43]
v_mfma_f32_16x16x32_f16 a[40:43], v[200:203], v[66:69], a[40:43]
v_mfma_f32_16x16x32_f16 a[44:47], v[172:175], v[54:57], a[44:47]
v_mfma_f32_16x16x32_f16 a[44:47], v[176:179], v[58:61], a[44:47]
v_mfma_f32_16x16x32_f16 a[48:51], v[180:183], v[54:57], a[48:51]
v_mfma_f32_16x16x32_f16 a[48:51], v[184:187], v[58:61], a[48:51]
v_mfma_f32_16x16x32_f16 a[52:55], v[188:191], v[54:57], a[52:55]
v_mfma_f32_16x16x32_f16 a[52:55], v[192:195], v[58:61], a[52:55]
v_mfma_f32_16x16x32_f16 a[56:59], v[196:199], v[54:57], a[56:59]
v_mfma_f32_16x16x32_f16 a[56:59], v[200:203], v[58:61], a[56:59]
v_mfma_f32_16x16x32_f16 a[60:63], v[172:175], v[46:49], a[60:63]
v_mfma_f32_16x16x32_f16 a[60:63], v[176:179], v[50:53], a[60:63]
v_mfma_f32_16x16x32_f16 a[64:67], v[180:183], v[46:49], a[64:67]
v_mfma_f32_16x16x32_f16 a[64:67], v[184:187], v[50:53], a[64:67]
v_mfma_f32_16x16x32_f16 a[68:71], v[188:191], v[46:49], a[68:71]
v_mfma_f32_16x16x32_f16 a[68:71], v[192:195], v[50:53], a[68:71]
v_mfma_f32_16x16x32_f16 a[72:75], v[196:199], v[46:49], a[72:75]
v_mfma_f32_16x16x32_f16 a[72:75], v[200:203], v[50:53], a[72:75]
v_mfma_f32_16x16x32_f16 a[76:79], v[172:175], v[38:41], a[76:79]
v_mfma_f32_16x16x32_f16 a[76:79], v[176:179], v[42:45], a[76:79]
v_mfma_f32_16x16x32_f16 a[80:83], v[180:183], v[38:41], a[80:83]
v_mfma_f32_16x16x32_f16 a[80:83], v[184:187], v[42:45], a[80:83]
v_mfma_f32_16x16x32_f16 a[84:87], v[188:191], v[38:41], a[84:87]
v_mfma_f32_16x16x32_f16 a[84:87], v[192:195], v[42:45], a[84:87]
v_mfma_f32_16x16x32_f16 a[88:91], v[196:199], v[38:41], a[88:91]
v_mfma_f32_16x16x32_f16 a[88:91], v[200:203], v[42:45], a[88:91]
v_mfma_f32_16x16x32_f16 a[92:95], v[172:175], v[30:33], a[92:95]
v_mfma_f32_16x16x32_f16 a[92:95], v[176:179], v[34:37], a[92:95]
v_mfma_f32_16x16x32_f16 a[96:99], v[180:183], v[30:33], a[96:99]
v_mfma_f32_16x16x32_f16 a[96:99], v[184:187], v[34:37], a[96:99]
v_mfma_f32_16x16x32_f16 a[100:103], v[188:191], v[30:33], a[100:103]
v_mfma_f32_16x16x32_f16 a[100:103], v[192:195], v[34:37], a[100:103]
v_mfma_f32_16x16x32_f16 a[104:107], v[196:199], v[30:33], a[104:107]
v_mfma_f32_16x16x32_f16 a[104:107], v[200:203], v[34:37], a[104:107]
v_mfma_f32_16x16x32_f16 a[108:111], v[172:175], v[22:25], a[108:111]
v_mfma_f32_16x16x32_f16 a[108:111], v[176:179], v[26:29], a[108:111]
v_mfma_f32_16x16x32_f16 a[112:115], v[180:183], v[22:25], a[112:115]
v_mfma_f32_16x16x32_f16 a[112:115], v[184:187], v[26:29], a[112:115]
v_mfma_f32_16x16x32_f16 a[116:119], v[188:191], v[22:25], a[116:119]
v_mfma_f32_16x16x32_f16 a[116:119], v[192:195], v[26:29], a[116:119]
v_mfma_f32_16x16x32_f16 a[120:123], v[196:199], v[22:25], a[120:123]
v_mfma_f32_16x16x32_f16 a[120:123], v[200:203], v[26:29], a[120:123]
v_accvgpr_read_b32 v4, a124
v_accvgpr_read_b32 v5, a125
v_cvt_pk_f16_f32 v8, v4, v5
v_accvgpr_read_b32 v4, a126
v_accvgpr_read_b32 v5, a127
v_cvt_pk_f16_f32 v9, v4, v5
v_accvgpr_read_b32 v4, a0
v_accvgpr_read_b32 v5, a1
v_cvt_pk_f16_f32 v12, v4, v5
v_accvgpr_read_b32 v4, a2
v_accvgpr_read_b32 v5, a3
v_cvt_pk_f16_f32 v13, v4, v5
v_accvgpr_read_b32 v4, a4
v_accvgpr_read_b32 v5, a5
v_cvt_pk_f16_f32 v16, v4, v5
v_accvgpr_read_b32 v4, a6
v_accvgpr_read_b32 v5, a7
v_cvt_pk_f16_f32 v17, v4, v5
v_accvgpr_read_b32 v4, a8
v_accvgpr_read_b32 v5, a9
v_cvt_pk_f16_f32 v20, v4, v5
v_accvgpr_read_b32 v4, a10
v_accvgpr_read_b32 v5, a11
v_cvt_pk_f16_f32 v21, v4, v5
v_accvgpr_read_b32 v4, a12
v_accvgpr_read_b32 v5, a13
v_cvt_pk_f16_f32 v10, v4, v5
v_accvgpr_read_b32 v4, a14
v_accvgpr_read_b32 v5, a15
v_cvt_pk_f16_f32 v11, v4, v5
v_accvgpr_read_b32 v4, a16
v_accvgpr_read_b32 v5, a17
v_cvt_pk_f16_f32 v14, v4, v5
v_accvgpr_read_b32 v4, a18
v_accvgpr_read_b32 v5, a19
v_cvt_pk_f16_f32 v15, v4, v5
v_accvgpr_read_b32 v4, a20
v_accvgpr_read_b32 v5, a21
v_cvt_pk_f16_f32 v18, v4, v5
v_accvgpr_read_b32 v4, a22
v_accvgpr_read_b32 v5, a23
v_cvt_pk_f16_f32 v19, v4, v5
v_accvgpr_read_b32 v4, a24
v_accvgpr_read_b32 v5, a25
v_cvt_pk_f16_f32 v22, v4, v5
v_accvgpr_read_b32 v4, a26
v_accvgpr_read_b32 v5, a27
v_cvt_pk_f16_f32 v23, v4, v5
v_accvgpr_read_b32 v4, a28
v_accvgpr_read_b32 v5, a29
v_cvt_pk_f16_f32 v24, v4, v5
v_accvgpr_read_b32 v4, a30
v_accvgpr_read_b32 v5, a31
v_cvt_pk_f16_f32 v25, v4, v5
v_accvgpr_read_b32 v4, a32
v_accvgpr_read_b32 v5, a33
v_cvt_pk_f16_f32 v28, v4, v5
v_accvgpr_read_b32 v4, a34
v_accvgpr_read_b32 v5, a35
v_cvt_pk_f16_f32 v29, v4, v5
v_accvgpr_read_b32 v4, a36
v_accvgpr_read_b32 v5, a37
v_cvt_pk_f16_f32 v32, v4, v5
v_accvgpr_read_b32 v4, a38
v_accvgpr_read_b32 v5, a39
v_cvt_pk_f16_f32 v33, v4, v5
v_accvgpr_read_b32 v4, a40
v_accvgpr_read_b32 v5, a41
v_cvt_pk_f16_f32 v36, v4, v5
v_accvgpr_read_b32 v4, a42
v_accvgpr_read_b32 v5, a43
v_cvt_pk_f16_f32 v37, v4, v5
v_accvgpr_read_b32 v4, a44
v_accvgpr_read_b32 v5, a45
v_cvt_pk_f16_f32 v26, v4, v5
v_accvgpr_read_b32 v4, a46
v_accvgpr_read_b32 v5, a47
v_cvt_pk_f16_f32 v27, v4, v5
v_accvgpr_read_b32 v4, a48
v_accvgpr_read_b32 v5, a49
v_cvt_pk_f16_f32 v30, v4, v5
v_accvgpr_read_b32 v4, a50
v_accvgpr_read_b32 v5, a51
v_cvt_pk_f16_f32 v31, v4, v5
v_accvgpr_read_b32 v4, a52
v_accvgpr_read_b32 v5, a53
v_cvt_pk_f16_f32 v34, v4, v5
v_accvgpr_read_b32 v4, a54
v_accvgpr_read_b32 v5, a55
v_cvt_pk_f16_f32 v35, v4, v5
v_accvgpr_read_b32 v4, a56
v_accvgpr_read_b32 v5, a57
v_cvt_pk_f16_f32 v38, v4, v5
v_accvgpr_read_b32 v4, a58
v_accvgpr_read_b32 v5, a59
v_cvt_pk_f16_f32 v39, v4, v5
v_accvgpr_read_b32 v4, a60
v_accvgpr_read_b32 v5, a61
v_cvt_pk_f16_f32 v40, v4, v5
v_accvgpr_read_b32 v4, a62
v_accvgpr_read_b32 v5, a63
v_cvt_pk_f16_f32 v41, v4, v5
v_accvgpr_read_b32 v4, a64
v_accvgpr_read_b32 v5, a65
v_cvt_pk_f16_f32 v44, v4, v5
v_accvgpr_read_b32 v4, a66
v_accvgpr_read_b32 v5, a67
v_cvt_pk_f16_f32 v45, v4, v5
v_accvgpr_read_b32 v4, a68
v_accvgpr_read_b32 v5, a69
v_cvt_pk_f16_f32 v48, v4, v5
v_accvgpr_read_b32 v4, a70
v_accvgpr_read_b32 v5, a71
v_cvt_pk_f16_f32 v49, v4, v5
v_accvgpr_read_b32 v4, a72
v_accvgpr_read_b32 v5, a73
v_cvt_pk_f16_f32 v52, v4, v5
v_accvgpr_read_b32 v4, a74
v_accvgpr_read_b32 v5, a75
v_cvt_pk_f16_f32 v53, v4, v5
v_accvgpr_read_b32 v4, a76
v_accvgpr_read_b32 v5, a77
v_cvt_pk_f16_f32 v42, v4, v5
v_accvgpr_read_b32 v4, a78
v_accvgpr_read_b32 v5, a79
v_cvt_pk_f16_f32 v43, v4, v5
v_accvgpr_read_b32 v4, a80
v_accvgpr_read_b32 v5, a81
v_cvt_pk_f16_f32 v46, v4, v5
v_accvgpr_read_b32 v4, a82
v_accvgpr_read_b32 v5, a83
v_cvt_pk_f16_f32 v47, v4, v5
v_accvgpr_read_b32 v4, a84
v_accvgpr_read_b32 v5, a85
v_cvt_pk_f16_f32 v50, v4, v5
v_accvgpr_read_b32 v4, a86
v_accvgpr_read_b32 v5, a87
v_cvt_pk_f16_f32 v51, v4, v5
v_accvgpr_read_b32 v4, a88
v_accvgpr_read_b32 v5, a89
v_cvt_pk_f16_f32 v54, v4, v5
v_accvgpr_read_b32 v4, a90
v_accvgpr_read_b32 v5, a91
v_cvt_pk_f16_f32 v55, v4, v5
v_accvgpr_read_b32 v4, a92
v_accvgpr_read_b32 v5, a93
v_cvt_pk_f16_f32 v56, v4, v5
v_accvgpr_read_b32 v4, a94
v_accvgpr_read_b32 v5, a95
v_cvt_pk_f16_f32 v57, v4, v5
v_accvgpr_read_b32 v4, a96
v_accvgpr_read_b32 v5, a97
v_cvt_pk_f16_f32 v60, v4, v5
v_accvgpr_read_b32 v4, a98
v_accvgpr_read_b32 v5, a99
v_cvt_pk_f16_f32 v61, v4, v5
v_accvgpr_read_b32 v4, a100
v_accvgpr_read_b32 v5, a101
v_cvt_pk_f16_f32 v64, v4, v5
v_accvgpr_read_b32 v4, a102
v_accvgpr_read_b32 v5, a103
v_cvt_pk_f16_f32 v65, v4, v5
v_accvgpr_read_b32 v4, a104
v_accvgpr_read_b32 v5, a105
v_cvt_pk_f16_f32 v68, v4, v5
v_accvgpr_read_b32 v4, a106
v_accvgpr_read_b32 v5, a107
v_cvt_pk_f16_f32 v69, v4, v5
v_accvgpr_read_b32 v4, a108
v_accvgpr_read_b32 v5, a109
v_cvt_pk_f16_f32 v58, v4, v5
v_accvgpr_read_b32 v4, a110
v_accvgpr_read_b32 v5, a111
v_cvt_pk_f16_f32 v59, v4, v5
v_accvgpr_read_b32 v4, a112
v_accvgpr_read_b32 v5, a113
v_cvt_pk_f16_f32 v62, v4, v5
v_accvgpr_read_b32 v4, a114
v_accvgpr_read_b32 v5, a115
v_cvt_pk_f16_f32 v63, v4, v5
v_accvgpr_read_b32 v4, a116
v_accvgpr_read_b32 v5, a117
v_cvt_pk_f16_f32 v66, v4, v5
v_accvgpr_read_b32 v4, a118
v_accvgpr_read_b32 v5, a119
v_cvt_pk_f16_f32 v67, v4, v5
v_accvgpr_read_b32 v4, a120
v_accvgpr_read_b32 v5, a121
v_cvt_pk_f16_f32 v70, v4, v5
v_accvgpr_read_b32 v4, a122
v_accvgpr_read_b32 v5, a123
v_cvt_pk_f16_f32 v71, v4, v5
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[8:11]
ds_write_b128 v2, v[12:15]
ds_write_b128 v3, v[16:19]
ds_write_b128 v6, v[20:23]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[8:11], v1
ds_read_b128 v[14:17], v1, offset:256
ds_read_b128 v[18:21], v1, offset:128
ds_read_b128 v[72:75], v1, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[24:27]
ds_write_b128 v2, v[28:31]
ds_write_b128 v3, v[32:35]
ds_write_b128 v6, v[36:39]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[24:27], v1
ds_read_b128 v[30:33], v1, offset:256
ds_read_b128 v[34:37], v1, offset:128
ds_read_b128 v[76:79], v1, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[40:43]
ds_write_b128 v2, v[44:47]
ds_write_b128 v3, v[48:51]
ds_write_b128 v6, v[52:55]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[40:43], v1
ds_read_b128 v[46:49], v1, offset:256
ds_read_b128 v[50:53], v1, offset:128
ds_read_b128 v[80:83], v1, offset:384
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v0, v[56:59]
ds_write_b128 v2, v[60:63]
ds_write_b128 v3, v[64:67]
ds_write_b128 v6, v[68:71]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[56:59], v1
ds_read_b128 v[2:5], v1, offset:256
ds_read_b128 v[60:63], v1, offset:128
ds_read_b128 v[66:69], v1, offset:384
v_mov_b32_e32 v12, v8
v_mov_b32_e32 v13, v9
buffer_store_dwordx4 v[12:15], v86, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v70, v18
v_mov_b32_e32 v71, v19
buffer_store_dwordx4 v[70:73], v87, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v12, v16
v_mov_b32_e32 v13, v17
buffer_store_dwordx4 v[10:13], v88, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v22, v74
v_mov_b32_e32 v23, v75
buffer_store_dwordx4 v[20:23], v89, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v28, v24
v_mov_b32_e32 v29, v25
buffer_store_dwordx4 v[28:31], v90, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v74, v34
v_mov_b32_e32 v75, v35
buffer_store_dwordx4 v[74:77], v91, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v28, v32
v_mov_b32_e32 v29, v33
buffer_store_dwordx4 v[26:29], v92, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v38, v78
v_mov_b32_e32 v39, v79
buffer_store_dwordx4 v[36:39], v93, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v44, v40
v_mov_b32_e32 v45, v41
buffer_store_dwordx4 v[44:47], v94, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v78, v50
v_mov_b32_e32 v79, v51
buffer_store_dwordx4 v[78:81], v95, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v44, v48
v_mov_b32_e32 v45, v49
buffer_store_dwordx4 v[42:45], v96, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v54, v82
v_mov_b32_e32 v55, v83
buffer_store_dwordx4 v[52:55], v97, s[0:3], 0, offen, offset:256
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v0, v56
v_mov_b32_e32 v1, v57
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[0:3], v98, s[0:3], 0, offen, offset:256
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v64, v60
v_mov_b32_e32 v65, v61
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[64:67], v99, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v60, v4
v_mov_b32_e32 v61, v5
buffer_store_dwordx4 v[58:61], v7, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v64, v68
v_mov_b32_e32 v65, v69
buffer_store_dwordx4 v[62:65], v100, s[0:3], 0, offen, offset:256
s_endpgm
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v7_slice
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
		.amdhsa_next_free_sgpr 54
		.amdhsa_accum_offset 204
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
	.size	v7_slice, .Lfunc_end0-v7_slice
	.cfi_endproc
                                        ; -- End function
	.set v7_slice.num_vgpr, 204
	.set v7_slice.num_agpr, 256
	.set v7_slice.numbered_sgpr, 54
	.set v7_slice.num_named_barrier, 0
	.set v7_slice.private_seg_size, 0
	.set v7_slice.uses_vcc, 0
	.set v7_slice.uses_flat_scratch, 0
	.set v7_slice.has_dyn_sized_stack, 0
	.set v7_slice.has_recursion, 0
	.set v7_slice.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14572
; TotalNumSgprs: 60
; NumVgprs: 204
; NumAgprs: 256
; TotalNumVgprs: 460
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 57
; NumSGPRsForWavesPerEU: 60
; NumVGPRsForWavesPerEU: 460
; AccumOffset: 204
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 50
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
	.asciz	"/var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v7_slice" ; string offset=24 ; /var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v7_slice
.Linfo_string3:
	.asciz	"v7_slice"                      ; string offset=91 ; v7_slice
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
    .name:           v7_slice
    .private_segment_fixed_size: 0
    .sgpr_count:     60
    .sgpr_spill_count: 0
    .symbol:         v7_slice.kd
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