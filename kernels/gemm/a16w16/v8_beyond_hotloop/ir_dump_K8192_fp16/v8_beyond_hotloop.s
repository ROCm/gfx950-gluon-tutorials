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
	.file	1 "/var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v8_beyond_hotloop" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.4:
.LBB0_0:
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s8, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_add_i32 s1, s9, 0xff
	s_ashr_i32 s8, s1, 31
	s_lshr_b32 s8, s8, 24
	s_add_i32 s1, s1, s8
	s_ashr_i32 s1, s1, 8
	s_lshl_b32 s9, s1, 2
	s_abs_i32 s14, s9
	v_mov_b32_e32 v18, v0
	v_cvt_f32_u32_e32 v0, s14
	s_ashr_i32 s8, s16, 31
	s_lshr_b32 s8, s8, 29
	s_add_i32 s8, s16, s8
	v_rcp_iflag_f32_e32 v0, v0
	s_ashr_i32 s8, s8, 3
	s_lshl_b32 s15, s16, 5
	s_sub_i32 s16, 0, s14
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_mulk_i32 s8, 0xff01
	s_add_i32 s8, s8, s15
	s_abs_i32 s15, s8
	v_readfirstlane_b32 s17, v0
	s_mul_i32 s16, s16, s17
	s_mul_hi_u32 s16, s17, s16
	s_add_i32 s17, s17, s16
	s_mul_hi_u32 s16, s15, s17
	s_mul_i32 s17, s16, s14
	v_readfirstlane_b32 s13, v18
	s_xor_b32 s1, s8, s1
	s_sub_i32 s15, s15, s17
	s_bfe_u32 s21, s13, 0x20006
	s_ashr_i32 s0, s0, 8
	s_ashr_i32 s1, s1, 31
	s_add_i32 s17, s16, 1
	s_sub_i32 s18, s15, s14
	s_cmp_ge_u32 s15, s14
	s_cselect_b32 s16, s17, s16
	s_cselect_b32 s15, s18, s15
	s_add_i32 s17, s16, 1
	s_cmp_ge_u32 s15, s14
	s_cselect_b32 s14, s17, s16
	s_xor_b32 s14, s14, s1
	s_sub_i32 s1, s14, s1
	s_lshl_b32 s14, s1, 2
	s_sub_i32 s0, s0, s14
	s_min_i32 s0, s0, 4
	s_abs_i32 s15, s0
	v_cvt_f32_u32_e32 v0, s15
	s_sub_i32 s16, 0, s15
	s_mul_i32 s1, s1, s9
	s_sub_i32 s1, s8, s1
	v_rcp_iflag_f32_e32 v0, v0
	s_abs_i32 s9, s1
	s_xor_b32 s8, s1, s0
	s_ashr_i32 s8, s8, 31
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_mul_i32 s50, s21, 0x420
	s_mov_b32 s19, 0x27000
	s_mov_b32 s23, s19
	v_readfirstlane_b32 s17, v0
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
	s_lshl_b32 s9, s0, 8
	s_mul_i32 s14, s9, s10
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[14:15], s[14:15], 1
	s_add_u32 s16, s2, s14
	v_and_b32_e32 v0, 63, v18
	s_addc_u32 s24, s3, s15
	s_lshl_b32 s8, s8, 8
	v_lshl_or_b32 v19, s21, 6, v0
	v_lshlrev_b32_e32 v0, 1, v18
	s_mul_i32 s14, s8, s11
	v_and_b32_e32 v0, 0x70, v0
	s_ashr_i32 s15, s14, 31
	v_or_b32_e32 v1, s21, v0
	v_lshlrev_b32_e32 v0, 3, v18
	s_lshl_b64 s[14:15], s[14:15], 1
	v_or_b32_e32 v4, 4, v1
	v_or_b32_e32 v6, 8, v1
	v_or_b32_e32 v8, 12, v1
	v_or_b32_e32 v2, 0x80, v1
	v_or_b32_e32 v3, 0x84, v1
	v_or_b32_e32 v5, 0x88, v1
	v_or_b32_e32 v7, 0x8c, v1
	v_accvgpr_write_b32 a129, v0
	v_and_b32_e32 v0, 56, v0
	s_add_u32 s20, s4, s14
	s_addc_u32 s1, s5, s15
	v_mul_lo_u32 v11, v4, s10
	v_mul_lo_u32 v12, v6, s10
	v_mul_lo_u32 v13, v8, s10
	v_mul_lo_u32 v14, v2, s10
	v_mul_lo_u32 v15, v3, s10
	v_mul_lo_u32 v16, v5, s10
	v_mul_lo_u32 v17, v7, s10
	v_mad_u64_u32 v[2:3], s[4:5], v1, s11, v[0:1]
	v_mad_u64_u32 v[4:5], s[4:5], v4, s11, v[0:1]
	v_mad_u64_u32 v[6:7], s[4:5], v6, s11, v[0:1]
	v_mad_u64_u32 v[8:9], s[4:5], v8, s11, v[0:1]
	s_lshl_b32 s4, s11, 8
	v_mul_lo_u32 v10, v1, s10
	s_ashr_i32 s5, s4, 1
	s_add_i32 s4, s50, 0
	s_and_b32 s17, s24, 0xffff
	s_mov_b32 s18, 0x7ffffffe
	v_add_lshl_u32 v1, v10, v0, 1
	s_mov_b32 m0, s4
	s_add_i32 s11, s4, 0x1080
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_add_lshl_u32 v3, v11, v0, 1
	s_mov_b32 m0, s11
	s_add_i32 s14, s4, 0x2100
	buffer_load_dwordx4 v3, s[16:19], 0 offen lds
	v_add_lshl_u32 v5, v12, v0, 1
	s_mov_b32 m0, s14
	s_add_i32 s15, s4, 0x3180
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	v_add_lshl_u32 v7, v13, v0, 1
	s_mov_b32 m0, s15
	s_add_i32 s28, s4, 0x4200
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	v_add_lshl_u32 v9, v14, v0, 1
	s_mov_b32 m0, s28
	s_add_i32 s29, s4, 0x5280
	buffer_load_dwordx4 v9, s[16:19], 0 offen lds
	v_add_lshl_u32 v10, v15, v0, 1
	s_mov_b32 m0, s29
	s_add_i32 s30, s4, 0x6300
	buffer_load_dwordx4 v10, s[16:19], 0 offen lds
	v_add_lshl_u32 v11, v16, v0, 1
	s_mov_b32 m0, s30
	s_add_i32 s31, s4, 0x7380
	s_add_i32 s54, 0, 0x107e0
	s_add_i32 s51, s50, 0x1080
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	v_add_lshl_u32 v0, v17, v0, 1
	s_mov_b32 m0, s31
	s_add_i32 s33, s54, s50
	s_add_i32 s52, s50, 0x2100
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	s_and_b32 s21, s1, 0xffff
	s_mov_b32 s22, s18
	v_lshlrev_b32_e32 v12, 1, v2
	s_mov_b32 m0, s33
	s_add_i32 s34, s54, s51
	s_add_i32 s53, s50, 0x3180
	buffer_load_dwordx4 v12, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v13, 1, v4
	s_mov_b32 m0, s34
	s_add_i32 s35, s54, s52
	buffer_load_dwordx4 v13, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v14, 1, v6
	s_mov_b32 m0, s35
	s_add_i32 s36, s54, s53
	s_add_i32 s40, 0, 0x18bc0
	buffer_load_dwordx4 v14, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v8
	s_mov_b32 m0, s36
	s_add_i32 s37, s40, s50
	buffer_load_dwordx4 v15, s[20:23], 0 offen lds
	v_add_lshl_u32 v2, v2, s5, 1
	s_mov_b32 m0, s37
	s_add_i32 s38, s40, s51
	s_add_i32 s39, s40, s52
	s_add_i32 s40, s40, s53
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	v_add_lshl_u32 v4, v4, s5, 1
	s_mov_b32 m0, s38
	s_add_u32 s16, s16, 0x80
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	v_add_lshl_u32 v6, v6, s5, 1
	s_mov_b32 m0, s39
	v_add_lshl_u32 v8, v8, s5, 1
	s_addc_u32 s5, s24, 0
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	s_mov_b32 m0, s40
	s_add_u32 s24, s20, 0x80
	buffer_load_dwordx4 v8, s[20:23], 0 offen lds
	s_addc_u32 s25, s1, 0
	s_add_i32 s21, s37, 0xfffef840
	s_and_b32 s17, s5, 0xffff
	s_mov_b32 m0, s21
	s_add_i32 s22, s37, 0xffff08c0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	s_mov_b32 m0, s22
	s_add_i32 s23, s37, 0xffff1940
	buffer_load_dwordx4 v3, s[16:19], 0 offen lds
	s_mov_b32 m0, s23
	s_add_i32 s41, s37, 0xffff29c0
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s42, s37, 0xffff3a40
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_mov_b32 m0, s42
	s_add_i32 s43, s37, 0xffff4ac0
	buffer_load_dwordx4 v9, s[16:19], 0 offen lds
	s_mov_b32 m0, s43
	s_add_i32 s44, s37, 0xffff5b40
	buffer_load_dwordx4 v10, s[16:19], 0 offen lds
	s_mov_b32 m0, s44
	s_add_i32 s45, s37, 0xffff6bc0
	s_add_i32 s49, 0, 0x149e0
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s46, s49, s50
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	s_and_b32 s25, s25, 0xffff
	s_mov_b32 s26, s18
	s_mov_b32 s27, s19
	s_mov_b32 m0, s46
	s_add_i32 s47, s49, s51
	buffer_load_dwordx4 v12, s[24:27], 0 offen lds
	s_mov_b32 m0, s47
	s_add_i32 s48, s49, s52
	buffer_load_dwordx4 v13, s[24:27], 0 offen lds
	s_mov_b32 m0, s48
	s_add_i32 s49, s49, s53
	s_add_i32 s5, 0, 0x1cdc0
	buffer_load_dwordx4 v14, s[24:27], 0 offen lds
	s_mov_b32 m0, s49
	s_add_i32 s50, s5, s50
	buffer_load_dwordx4 v15, s[24:27], 0 offen lds
	s_mov_b32 m0, s50
	s_add_i32 s51, s5, s51
	buffer_load_dwordx4 v2, s[24:27], 0 offen lds
	s_mov_b32 m0, s51
	s_add_i32 s52, s5, s52
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	s_mov_b32 m0, s52
	s_add_i32 s53, s5, s53
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	s_mov_b32 m0, s53
	v_accvgpr_write_b32 a111, v2
	buffer_load_dwordx4 v8, s[24:27], 0 offen lds
	v_and_b32_e32 v2, 15, v18
	v_accvgpr_write_b32 a105, v0
	v_lshlrev_b32_e32 v0, 10, v2
	s_movk_i32 s5, 0xb0
	v_accvgpr_write_b32 a97, v1
	v_and_or_b32 v1, v19, s5, v0
	v_accvgpr_write_b32 a115, v2
	v_lshlrev_b32_e32 v2, 5, v2
	v_add_u32_e32 v1, v1, v2
	v_add_u32_e32 v1, 0, v1
	s_waitcnt vmcnt(20) lgkmcnt(0)
	s_barrier
	ds_read_b128 a[56:59], v1
	ds_read_b128 a[60:63], v1 offset:64
	ds_read_b128 a[48:51], v1 offset:256
	ds_read_b128 a[52:55], v1 offset:320
	ds_read_b128 a[40:43], v1 offset:512
	ds_read_b128 a[44:47], v1 offset:576
	ds_read_b128 a[32:35], v1 offset:768
	ds_read_b128 a[36:39], v1 offset:832
	ds_read_b128 a[24:27], v1 offset:16896
	ds_read_b128 a[28:31], v1 offset:16960
	ds_read_b128 a[16:19], v1 offset:17152
	ds_read_b128 a[20:23], v1 offset:17216
	ds_read_b128 a[8:11], v1 offset:17408
	ds_read_b128 a[12:15], v1 offset:17472
	ds_read_b128 a[0:3], v1 offset:17664
	v_and_or_b32 v0, v18, 48, v0
	s_and_b32 s5, s13, 64
	v_add_u32_e32 v0, v0, v2
	v_lshl_add_u32 v0, s5, 1, v0
	v_accvgpr_write_b32 a99, v1
	ds_read_b128 a[4:7], v1 offset:17728
	v_add_u32_e32 v1, s54, v0
	ds_read_b128 a[64:67], v1
	ds_read_b128 a[68:71], v1 offset:64
	ds_read_b128 a[72:75], v1 offset:256
	ds_read_b128 a[76:79], v1 offset:320
	ds_read_b128 a[80:83], v1 offset:512
	ds_read_b128 a[84:87], v1 offset:576
	ds_read_b128 a[88:91], v1 offset:768
	ds_read_b128 a[92:95], v1 offset:832
	s_add_u32 s20, s20, 0x180
	s_mul_i32 s10, s10, s0
	s_addc_u32 s24, s1, 0
	s_lshl_b32 s0, s10, 8
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_add_u32_e32 v0, 0, v0
	s_add_u32 s0, s2, s0
	v_mov_b32_e32 v24, 0
	v_add_u32_e32 v1, 0x149e0, v0
	v_accvgpr_write_b32 a100, v5
	s_addc_u32 s1, s3, s1
	v_mov_b32_e32 v5, v24
	v_accvgpr_write_b32 a158, v1
	v_add_u32_e32 v1, 0x18bc0, v0
	v_accvgpr_write_b32 a98, v3
	v_accvgpr_write_b32 a112, v4
	s_add_u32 s10, s0, 0x180
	v_mov_b32_e32 v2, v24
	v_mov_b32_e32 v3, v24
	v_mov_b32_e32 v4, v24
	v_accvgpr_write_b32 a165, v5
	v_accvgpr_write_b32 a119, v5
	v_accvgpr_write_b32 a123, v5
	v_accvgpr_write_b32 a127, v5
	v_accvgpr_write_b32 a133, v5
	v_accvgpr_write_b32 a137, v5
	v_accvgpr_write_b32 a141, v5
	v_accvgpr_write_b32 a145, v5
	v_accvgpr_write_b32 a149, v5
	v_accvgpr_write_b32 a153, v5
	v_accvgpr_write_b32 a157, v5
	v_accvgpr_write_b32 a159, v1
	v_add_u32_e32 v1, 0x1cdc0, v0
	v_accvgpr_write_b32 a108, v0
	v_add_u32_e32 v0, 0x107e0, v0
	v_accvgpr_write_b32 a101, v7
	v_accvgpr_write_b32 a102, v9
	v_accvgpr_write_b32 a103, v10
	v_accvgpr_write_b32 a104, v11
	v_accvgpr_write_b32 a106, v12
	v_accvgpr_write_b32 a107, v13
	v_accvgpr_write_b32 a109, v14
	v_accvgpr_write_b32 a110, v15
	v_accvgpr_write_b32 a113, v6
	v_accvgpr_write_b32 a114, v8
	v_accvgpr_write_b32 a96, v19
	v_accvgpr_write_b32 a128, v18
	s_addc_u32 s25, s1, 0
	s_mov_b32 s26, -2
	v_mov_b32_e32 v25, v24
	v_mov_b32_e32 v26, v24
	v_mov_b32_e32 v27, v24
	v_mov_b32_e32 v28, v24
	v_mov_b32_e32 v29, v24
	v_mov_b32_e32 v30, v24
	v_mov_b32_e32 v31, v24
	v_mov_b32_e32 v44, v24
	v_mov_b32_e32 v45, v24
	v_mov_b32_e32 v46, v24
	v_mov_b32_e32 v47, v24
	v_accvgpr_write_b32 a164, v4
	v_accvgpr_write_b32 a163, v3
	v_accvgpr_write_b32 a162, v2
	v_mov_b32_e32 v48, v24
	v_mov_b32_e32 v49, v24
	v_mov_b32_e32 v50, v24
	v_mov_b32_e32 v51, v24
	v_mov_b32_e32 v52, v24
	v_mov_b32_e32 v53, v24
	v_mov_b32_e32 v54, v24
	v_mov_b32_e32 v55, v24
	v_mov_b32_e32 v56, v24
	v_mov_b32_e32 v57, v24
	v_mov_b32_e32 v58, v24
	v_mov_b32_e32 v59, v24
	v_mov_b32_e32 v34, v24
	v_mov_b32_e32 v35, v24
	v_mov_b32_e32 v36, v24
	v_mov_b32_e32 v37, v24
	v_mov_b32_e32 v38, v24
	v_mov_b32_e32 v39, v24
	v_mov_b32_e32 v40, v24
	v_mov_b32_e32 v41, v24
	v_mov_b32_e32 v68, v24
	v_mov_b32_e32 v69, v24
	v_mov_b32_e32 v70, v24
	v_mov_b32_e32 v71, v24
	v_accvgpr_write_b32 a118, v4
	v_accvgpr_write_b32 a117, v3
	v_accvgpr_write_b32 a116, v2
	v_accvgpr_write_b32 a122, v4
	v_accvgpr_write_b32 a121, v3
	v_accvgpr_write_b32 a120, v2
	v_accvgpr_write_b32 a126, v4
	v_accvgpr_write_b32 a125, v3
	v_accvgpr_write_b32 a124, v2
	v_accvgpr_write_b32 a132, v4
	v_accvgpr_write_b32 a131, v3
	v_accvgpr_write_b32 a130, v2
	v_mov_b32_e32 v62, v24
	v_mov_b32_e32 v63, v24
	v_mov_b32_e32 v64, v24
	v_mov_b32_e32 v65, v24
	v_accvgpr_write_b32 a136, v4
	v_accvgpr_write_b32 a135, v3
	v_accvgpr_write_b32 a134, v2
	v_accvgpr_write_b32 a140, v4
	v_accvgpr_write_b32 a139, v3
	v_accvgpr_write_b32 a138, v2
	v_mov_b32_e32 v74, v24
	v_mov_b32_e32 v75, v24
	v_mov_b32_e32 v76, v24
	v_mov_b32_e32 v77, v24
	v_mov_b32_e32 v78, v24
	v_mov_b32_e32 v79, v24
	v_mov_b32_e32 v80, v24
	v_mov_b32_e32 v81, v24
	v_mov_b32_e32 v82, v24
	v_mov_b32_e32 v83, v24
	v_mov_b32_e32 v84, v24
	v_mov_b32_e32 v85, v24
	v_mov_b32_e32 v86, v24
	v_mov_b32_e32 v87, v24
	v_mov_b32_e32 v88, v24
	v_mov_b32_e32 v89, v24
	v_mov_b32_e32 v90, v24
	v_mov_b32_e32 v91, v24
	v_mov_b32_e32 v92, v24
	v_mov_b32_e32 v93, v24
	v_mov_b32_e32 v94, v24
	v_mov_b32_e32 v95, v24
	v_mov_b32_e32 v96, v24
	v_mov_b32_e32 v97, v24
	v_mov_b32_e32 v98, v24
	v_mov_b32_e32 v99, v24
	v_mov_b32_e32 v100, v24
	v_mov_b32_e32 v101, v24
	v_mov_b32_e32 v102, v24
	v_mov_b32_e32 v103, v24
	v_mov_b32_e32 v104, v24
	v_mov_b32_e32 v105, v24
	v_mov_b32_e32 v106, v24
	v_mov_b32_e32 v107, v24
	v_mov_b32_e32 v108, v24
	v_mov_b32_e32 v109, v24
	v_mov_b32_e32 v110, v24
	v_mov_b32_e32 v111, v24
	v_mov_b32_e32 v112, v24
	v_mov_b32_e32 v113, v24
	v_mov_b32_e32 v114, v24
	v_mov_b32_e32 v115, v24
	v_mov_b32_e32 v116, v24
	v_mov_b32_e32 v117, v24
	v_accvgpr_write_b32 a144, v4
	v_accvgpr_write_b32 a143, v3
	v_accvgpr_write_b32 a142, v2
	v_accvgpr_write_b32 a148, v4
	v_accvgpr_write_b32 a147, v3
	v_accvgpr_write_b32 a146, v2
	v_accvgpr_write_b32 a152, v4
	v_accvgpr_write_b32 a151, v3
	v_accvgpr_write_b32 a150, v2
	v_accvgpr_write_b32 a156, v4
	v_accvgpr_write_b32 a155, v3
	v_accvgpr_write_b32 a154, v2
	v_mov_b32_e32 v18, v24
	v_mov_b32_e32 v19, v24
	v_mov_b32_e32 v20, v24
	v_mov_b32_e32 v21, v24
	v_accvgpr_write_b32 a170, v24
	v_accvgpr_write_b32 a171, v24
	v_accvgpr_write_b32 a172, v24
	v_accvgpr_write_b32 a173, v24
	v_accvgpr_write_b32 a204, v24
	v_accvgpr_write_b32 a205, v24
	v_accvgpr_write_b32 a206, v24
	v_accvgpr_write_b32 a207, v24
	v_mov_b32_e32 v138, v24
	v_mov_b32_e32 v139, v24
	v_mov_b32_e32 v140, v24
	v_mov_b32_e32 v141, v24
	v_mov_b32_e32 v142, v24
	v_mov_b32_e32 v143, v24
	v_mov_b32_e32 v144, v24
	v_mov_b32_e32 v145, v24
	v_mov_b32_e32 v146, v24
	v_mov_b32_e32 v147, v24
	v_mov_b32_e32 v148, v24
	v_mov_b32_e32 v149, v24
	v_mov_b32_e32 v150, v24
	v_mov_b32_e32 v151, v24
	v_mov_b32_e32 v152, v24
	v_mov_b32_e32 v153, v24
	v_mov_b32_e32 v154, v24
	v_mov_b32_e32 v155, v24
	v_mov_b32_e32 v156, v24
	v_mov_b32_e32 v157, v24
	v_mov_b32_e32 v162, v24
	v_mov_b32_e32 v163, v24
	v_mov_b32_e32 v164, v24
	v_mov_b32_e32 v165, v24
	v_mov_b32_e32 v166, v24
	v_mov_b32_e32 v167, v24
	v_mov_b32_e32 v168, v24
	v_mov_b32_e32 v169, v24
	v_mov_b32_e32 v170, v24
	v_mov_b32_e32 v171, v24
	v_mov_b32_e32 v172, v24
	v_mov_b32_e32 v173, v24
	v_mov_b32_e32 v174, v24
	v_mov_b32_e32 v175, v24
	v_mov_b32_e32 v176, v24
	v_mov_b32_e32 v177, v24
	v_mov_b32_e32 v178, v24
	v_mov_b32_e32 v179, v24
	v_mov_b32_e32 v180, v24
	v_mov_b32_e32 v181, v24
	v_mov_b32_e32 v182, v24
	v_mov_b32_e32 v183, v24
	v_mov_b32_e32 v184, v24
	v_mov_b32_e32 v185, v24
	v_mov_b32_e32 v186, v24
	v_mov_b32_e32 v187, v24
	v_mov_b32_e32 v188, v24
	v_mov_b32_e32 v189, v24
	v_mov_b32_e32 v190, v24
	v_mov_b32_e32 v191, v24
	v_mov_b32_e32 v192, v24
	v_mov_b32_e32 v193, v24
	v_mov_b32_e32 v194, v24
	v_mov_b32_e32 v195, v24
	v_mov_b32_e32 v196, v24
	v_mov_b32_e32 v197, v24
	v_mov_b32_e32 v198, v24
	v_mov_b32_e32 v199, v24
	v_mov_b32_e32 v200, v24
	v_mov_b32_e32 v201, v24
	v_mov_b32_e32 v202, v24
	v_mov_b32_e32 v203, v24
	v_mov_b32_e32 v204, v24
	v_mov_b32_e32 v205, v24
	v_mov_b32_e32 v206, v24
	v_mov_b32_e32 v207, v24
	v_mov_b32_e32 v208, v24
	v_mov_b32_e32 v209, v24
	v_mov_b32_e32 v210, v24
	v_mov_b32_e32 v211, v24
	v_mov_b32_e32 v212, v24
	v_mov_b32_e32 v213, v24
	v_mov_b32_e32 v214, v24
	v_mov_b32_e32 v215, v24
	v_mov_b32_e32 v216, v24
	v_mov_b32_e32 v217, v24
	v_mov_b32_e32 v218, v24
	v_mov_b32_e32 v219, v24
	v_mov_b32_e32 v220, v24
	v_mov_b32_e32 v221, v24
	v_mov_b32_e32 v222, v24
	v_mov_b32_e32 v223, v24
	v_mov_b32_e32 v224, v24
	v_mov_b32_e32 v225, v24
	v_mov_b32_e32 v226, v24
	v_mov_b32_e32 v227, v24
	v_mov_b32_e32 v228, v24
	v_mov_b32_e32 v229, v24
	v_mov_b32_e32 v230, v24
	v_mov_b32_e32 v231, v24
	v_mov_b32_e32 v232, v24
	v_mov_b32_e32 v233, v24
	v_mov_b32_e32 v234, v24
	v_mov_b32_e32 v235, v24
	v_mov_b32_e32 v236, v24
	v_mov_b32_e32 v237, v24
	v_mov_b32_e32 v238, v24
	v_mov_b32_e32 v239, v24
	v_mov_b32_e32 v240, v24
	v_mov_b32_e32 v241, v24
	v_mov_b32_e32 v242, v24
	v_mov_b32_e32 v243, v24
	v_mov_b32_e32 v244, v24
	v_mov_b32_e32 v245, v24
	v_mov_b32_e32 v246, v24
	v_mov_b32_e32 v247, v24
	v_mov_b32_e32 v248, v24
	v_mov_b32_e32 v249, v24
	v_mov_b32_e32 v250, v24
	v_mov_b32_e32 v251, v24
	v_mov_b32_e32 v252, v24
	v_mov_b32_e32 v253, v24
	v_mov_b32_e32 v6, v24
	v_mov_b32_e32 v7, v24
	v_accvgpr_write_b32 a160, v1
	v_accvgpr_write_b32 a161, v0
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[6:9], a[88:91], a[0:3], v[4:7]
	s_add_u32 s0, s20, 0xffffff80
	s_addc_u32 s1, s24, -1
	s_add_u32 s16, s10, 0xffffff80
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[4:7], v[6:9]
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[158:161], a[64:67], a[56:59], v[18:21]
	s_addc_u32 s2, s25, -1
	s_and_b32 s17, s2, 0xffff
	s_mov_b32 m0, s4
	v_mfma_f32_16x16x32_f16 a[166:169], a[72:75], a[56:59], a[170:173]
	s_nop 0
	v_accvgpr_write_b32 a199, v3
	v_accvgpr_write_b32 a198, v2
	v_accvgpr_write_b32 a197, v1
	v_mfma_f32_16x16x32_f16 v[142:145], a[64:67], a[48:51], v[142:145]
	v_accvgpr_write_b32 a196, v0
	v_accvgpr_read_b32 v0, a159
	s_and_b32 s1, s1, 0xffff
	v_mfma_f32_16x16x32_f16 v[146:149], a[72:75], a[48:51], v[146:149]
	s_mov_b32 s2, s18
	s_mov_b32 s3, s19
	v_accvgpr_read_b32 v9, a106
	v_mfma_f32_16x16x32_f16 v[162:165], a[64:67], a[40:43], v[162:165]
	v_accvgpr_read_b32 v254, a107
	v_accvgpr_read_b32 v255, a109
	v_accvgpr_read_b32 v8, a99
	v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[40:43], v[166:169]
	v_mfma_f32_16x16x32_f16 v[178:181], a[64:67], a[32:35], v[178:181]
	v_mfma_f32_16x16x32_f16 v[182:185], a[72:75], a[32:35], v[182:185]
	v_mfma_f32_16x16x32_f16 v[194:197], a[64:67], a[24:27], v[194:197]
	v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[24:27], v[198:201]
	v_mfma_f32_16x16x32_f16 v[210:213], a[64:67], a[16:19], v[210:213]
	v_mfma_f32_16x16x32_f16 v[214:217], a[72:75], a[16:19], v[214:217]
	v_mfma_f32_16x16x32_f16 v[226:229], a[64:67], a[8:11], v[226:229]
	v_mfma_f32_16x16x32_f16 v[230:233], a[72:75], a[8:11], v[230:233]
	v_mfma_f32_16x16x32_f16 v[242:245], a[64:67], a[0:3], v[242:245]
	v_mfma_f32_16x16x32_f16 v[246:249], a[72:75], a[0:3], v[246:249]
	v_mfma_f32_16x16x32_f16 v[158:161], a[68:71], a[60:63], v[158:161]
	v_mfma_f32_16x16x32_f16 a[200:203], a[76:79], a[60:63], a[166:169]
	v_mfma_f32_16x16x32_f16 a[166:169], a[80:83], a[56:59], a[204:207]
	v_mfma_f32_16x16x32_f16 v[142:145], a[68:71], a[52:55], v[142:145]
	v_mfma_f32_16x16x32_f16 v[146:149], a[76:79], a[52:55], v[146:149]
	v_mfma_f32_16x16x32_f16 v[150:153], a[80:83], a[48:51], v[150:153]
	v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[44:47], v[162:165]
	v_mfma_f32_16x16x32_f16 v[166:169], a[76:79], a[44:47], v[166:169]
	v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[40:43], v[170:173]
	v_mfma_f32_16x16x32_f16 v[178:181], a[68:71], a[36:39], v[178:181]
	v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[36:39], v[182:185]
	v_mfma_f32_16x16x32_f16 v[186:189], a[80:83], a[32:35], v[186:189]
	v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[28:31], v[194:197]
	v_mfma_f32_16x16x32_f16 v[198:201], a[76:79], a[28:31], v[198:201]
	v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[24:27], v[202:205]
	v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[20:23], v[210:213]
	v_mfma_f32_16x16x32_f16 v[214:217], a[76:79], a[20:23], v[214:217]
	v_mfma_f32_16x16x32_f16 v[218:221], a[80:83], a[16:19], v[218:221]
	v_mfma_f32_16x16x32_f16 v[226:229], a[68:71], a[12:15], v[226:229]
	v_mfma_f32_16x16x32_f16 v[230:233], a[76:79], a[12:15], v[230:233]
	v_mfma_f32_16x16x32_f16 v[234:237], a[80:83], a[8:11], v[234:237]
	v_mfma_f32_16x16x32_f16 v[242:245], a[68:71], a[4:7], v[242:245]
	v_mfma_f32_16x16x32_f16 v[246:249], a[76:79], a[4:7], v[246:249]
	v_mfma_f32_16x16x32_f16 v[250:253], a[80:83], a[0:3], v[250:253]
	ds_read_b128 v[118:121], v0
	ds_read_b128 v[122:125], v0 offset:64
	ds_read_b128 v[126:129], v0 offset:256
	ds_read_b128 a[64:67], v0 offset:320
	ds_read_b128 a[68:71], v0 offset:512
	ds_read_b128 a[72:75], v0 offset:576
	ds_read_b128 a[76:79], v0 offset:768
	ds_read_b128 a[80:83], v0 offset:832
	v_accvgpr_read_b32 v0, a97
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	s_mov_b32 m0, s11
	v_accvgpr_read_b32 v0, a98
	v_mfma_f32_16x16x32_f16 v[154:157], a[88:91], a[48:51], v[154:157]
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	s_mov_b32 m0, s14
	v_accvgpr_read_b32 v0, a100
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[2:5], v[118:121], a[56:59], v[24:27]
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	s_mov_b32 m0, s15
	v_accvgpr_read_b32 v0, a101
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[10:13], v[126:129], a[56:59], v[28:31]
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	s_mov_b32 m0, s28
	v_accvgpr_read_b32 v0, a102
	v_mfma_f32_16x16x32_f16 v[22:25], v[118:121], a[48:51], v[48:51]
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	s_mov_b32 m0, s29
	v_accvgpr_read_b32 v0, a103
	v_mfma_f32_16x16x32_f16 v[26:29], v[126:129], a[48:51], v[52:55]
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	s_mov_b32 m0, s30
	v_accvgpr_read_b32 v0, a104
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[30:33], a[68:71], a[48:51], v[56:59]
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	s_mov_b32 m0, s31
	v_accvgpr_read_b32 v0, a105
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[34:37], a[76:79], a[48:51], v[34:37]
	v_accvgpr_mov_b32 a48, a116
	v_accvgpr_mov_b32 a49, a117
	v_accvgpr_mov_b32 a50, a118
	v_accvgpr_mov_b32 a51, a119
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	s_mov_b32 m0, s33
	v_mfma_f32_16x16x32_f16 a[48:51], a[68:71], a[40:43], a[48:51]
	buffer_load_dwordx4 v9, s[0:3], 0 offen lds
	s_mov_b32 m0, s34
	v_accvgpr_read_b32 v0, a110
	v_mfma_f32_16x16x32_f16 a[116:119], a[72:75], a[44:47], a[48:51]
	buffer_load_dwordx4 v254, s[0:3], 0 offen lds
	s_mov_b32 m0, s35
	s_and_b32 s17, s25, 0xffff
	v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[40:43], v[174:177]
	v_accvgpr_mov_b32 a48, a120
	v_accvgpr_mov_b32 a49, a121
	v_accvgpr_mov_b32 a50, a122
	v_accvgpr_mov_b32 a51, a123
	v_mfma_f32_16x16x32_f16 v[14:17], a[68:71], a[56:59], v[44:47]
	buffer_load_dwordx4 v255, s[0:3], 0 offen lds
	s_mov_b32 m0, s36
	s_mov_b32 s16, s10
	v_mfma_f32_16x16x32_f16 v[38:41], v[118:121], a[40:43], v[38:41]
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	s_waitcnt vmcnt(16) lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[42:45], v[126:129], a[40:43], v[68:71]
	s_barrier
	s_mov_b32 m0, s37
	v_mfma_f32_16x16x32_f16 a[40:43], a[76:79], a[40:43], a[48:51]
	v_mfma_f32_16x16x32_f16 v[0:3], v[122:125], a[60:63], v[2:5]
	v_mfma_f32_16x16x32_f16 a[120:123], a[80:83], a[44:47], a[40:43]
	s_nop 1
	v_accvgpr_read_b32 v4, a158
	s_nop 2
	v_accvgpr_mov_b32 a40, a130
	v_accvgpr_mov_b32 a41, a131
	v_accvgpr_mov_b32 a42, a132
	v_accvgpr_mov_b32 a43, a133
	v_accvgpr_write_b32 a179, v3
	v_accvgpr_write_b32 a178, v2
	v_mfma_f32_16x16x32_f16 a[40:43], v[126:129], a[32:35], a[40:43]
	v_accvgpr_write_b32 a177, v1
	v_accvgpr_write_b32 a176, v0
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[60:63], v[14:17]
	v_mfma_f32_16x16x32_f16 a[130:133], a[64:67], a[36:39], a[40:43]
	s_nop 1
	v_accvgpr_read_b32 v14, a111
	v_accvgpr_read_b32 v15, a112
	v_accvgpr_read_b32 v16, a113
	v_accvgpr_mov_b32 a40, a134
	s_nop 0
	v_accvgpr_write_b32 a183, v3
	v_accvgpr_mov_b32 a41, a135
	v_accvgpr_mov_b32 a42, a136
	v_accvgpr_mov_b32 a43, a137
	v_mfma_f32_16x16x32_f16 v[190:193], a[88:91], a[32:35], v[190:193]
	v_accvgpr_write_b32 a182, v2
	v_accvgpr_write_b32 a181, v1
	v_accvgpr_write_b32 a180, v0
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[52:55], v[26:29]
	v_accvgpr_read_b32 v17, a114
	v_accvgpr_mov_b32 a170, a180
	v_accvgpr_mov_b32 a171, a181
	v_mfma_f32_16x16x32_f16 a[124:127], v[118:121], a[32:35], a[124:127]
	v_accvgpr_read_b32 v26, a160
	v_accvgpr_mov_b32 a172, a182
	v_accvgpr_mov_b32 a173, a183
	v_mfma_f32_16x16x32_f16 v[62:65], a[68:71], a[32:35], v[62:65]
	v_accvgpr_write_b32 a187, v3
	v_accvgpr_write_b32 a186, v2
	v_accvgpr_write_b32 a185, v1
	v_mfma_f32_16x16x32_f16 a[32:35], a[76:79], a[32:35], a[40:43]
	v_accvgpr_write_b32 a184, v0
	v_mfma_f32_16x16x32_f16 a[204:207], a[84:87], a[60:63], a[166:169]
	v_mfma_f32_16x16x32_f16 v[150:153], a[84:87], a[52:55], v[150:153]
	s_nop 1
	v_accvgpr_mov_b32 a166, a176
	v_accvgpr_mov_b32 a167, a177
	v_accvgpr_mov_b32 a168, a178
	v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[44:47], v[170:173]
	v_accvgpr_mov_b32 a169, a179
	v_mfma_f32_16x16x32_f16 v[186:189], a[84:87], a[36:39], v[186:189]
	v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[28:31], v[202:205]
	v_mfma_f32_16x16x32_f16 v[218:221], a[84:87], a[20:23], v[218:221]
	v_mfma_f32_16x16x32_f16 v[234:237], a[84:87], a[12:15], v[234:237]
	v_mfma_f32_16x16x32_f16 v[250:253], a[84:87], a[4:7], v[250:253]
	v_accvgpr_mov_b32 a84, a162
	v_accvgpr_mov_b32 a85, a163
	v_accvgpr_mov_b32 a86, a164
	v_mfma_f32_16x16x32_f16 a[134:137], a[80:83], a[36:39], a[32:35]
	v_accvgpr_mov_b32 a87, a165
	s_nop 1
	v_accvgpr_mov_b32 a32, a138
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[52:55], v[30:33]
	v_accvgpr_mov_b32 a33, a139
	v_accvgpr_mov_b32 a34, a140
	v_accvgpr_mov_b32 a35, a141
	v_mfma_f32_16x16x32_f16 v[138:141], a[88:91], a[56:59], v[138:141]
	v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[24:27], v[206:209]
	s_nop 2
	v_accvgpr_write_b32 a191, v3
	v_accvgpr_write_b32 a190, v2
	v_accvgpr_write_b32 a189, v1
	v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[16:19], v[222:225]
	v_accvgpr_write_b32 a188, v0
	v_mfma_f32_16x16x32_f16 v[238:241], a[88:91], a[8:11], v[238:241]
	v_mfma_f32_16x16x32_f16 a[56:59], a[76:79], a[56:59], a[84:87]
	v_mfma_f32_16x16x32_f16 a[32:35], v[118:121], a[24:27], a[32:35]
	v_mfma_f32_16x16x32_f16 v[74:77], v[126:129], a[24:27], v[74:77]
	v_mfma_f32_16x16x32_f16 v[78:81], a[68:71], a[24:27], v[78:81]
	v_mfma_f32_16x16x32_f16 v[82:85], a[76:79], a[24:27], v[82:85]
	v_mfma_f32_16x16x32_f16 v[86:89], v[118:121], a[16:19], v[86:89]
	v_mfma_f32_16x16x32_f16 v[90:93], v[126:129], a[16:19], v[90:93]
	v_mfma_f32_16x16x32_f16 v[94:97], a[68:71], a[16:19], v[94:97]
	v_mfma_f32_16x16x32_f16 v[98:101], a[76:79], a[16:19], v[98:101]
	v_mfma_f32_16x16x32_f16 v[102:105], v[118:121], a[8:11], v[102:105]
	v_mfma_f32_16x16x32_f16 v[106:109], v[126:129], a[8:11], v[106:109]
	v_mfma_f32_16x16x32_f16 v[110:113], a[68:71], a[8:11], v[110:113]
	v_mfma_f32_16x16x32_f16 v[114:117], a[76:79], a[8:11], v[114:117]
	v_mfma_f32_16x16x32_f16 a[142:145], v[118:121], a[0:3], a[142:145]
	v_mfma_f32_16x16x32_f16 a[146:149], v[126:129], a[0:3], a[146:149]
	v_mfma_f32_16x16x32_f16 a[150:153], a[68:71], a[0:3], a[150:153]
	v_mfma_f32_16x16x32_f16 a[154:157], a[76:79], a[0:3], a[154:157]
	v_mfma_f32_16x16x32_f16 v[138:141], a[92:95], a[60:63], v[138:141]
	v_mfma_f32_16x16x32_f16 v[154:157], a[92:95], a[52:55], v[154:157]
	v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[44:47], v[174:177]
	v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[36:39], v[190:193]
	v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[28:31], v[206:209]
	v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[20:23], v[222:225]
	v_mfma_f32_16x16x32_f16 v[238:241], a[92:95], a[12:15], v[238:241]
	v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[60:63], v[10:13]
	v_mfma_f32_16x16x32_f16 a[162:165], a[80:83], a[60:63], a[56:59]
	v_mfma_f32_16x16x32_f16 v[22:25], v[122:125], a[52:55], v[22:25]
	v_mfma_f32_16x16x32_f16 v[34:37], a[80:83], a[52:55], v[34:37]
	v_mfma_f32_16x16x32_f16 v[38:41], v[122:125], a[44:47], v[38:41]
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[44:47], v[42:45]
	v_mfma_f32_16x16x32_f16 a[124:127], v[122:125], a[36:39], a[124:127]
	v_mfma_f32_16x16x32_f16 v[62:65], a[72:75], a[36:39], v[62:65]
	v_mfma_f32_16x16x32_f16 a[138:141], v[122:125], a[28:31], a[32:35]
	v_mfma_f32_16x16x32_f16 v[74:77], a[64:67], a[28:31], v[74:77]
	v_mfma_f32_16x16x32_f16 v[78:81], a[72:75], a[28:31], v[78:81]
	v_mfma_f32_16x16x32_f16 v[82:85], a[80:83], a[28:31], v[82:85]
	v_mfma_f32_16x16x32_f16 v[86:89], v[122:125], a[20:23], v[86:89]
	v_mfma_f32_16x16x32_f16 v[90:93], a[64:67], a[20:23], v[90:93]
	v_mfma_f32_16x16x32_f16 v[94:97], a[72:75], a[20:23], v[94:97]
	v_mfma_f32_16x16x32_f16 v[98:101], a[80:83], a[20:23], v[98:101]
	v_mfma_f32_16x16x32_f16 v[102:105], v[122:125], a[12:15], v[102:105]
	v_mfma_f32_16x16x32_f16 v[106:109], a[64:67], a[12:15], v[106:109]
	v_mfma_f32_16x16x32_f16 v[110:113], a[72:75], a[12:15], v[110:113]
	v_mfma_f32_16x16x32_f16 v[114:117], a[80:83], a[12:15], v[114:117]
	v_mfma_f32_16x16x32_f16 a[142:145], v[122:125], a[4:7], a[142:145]
	v_mfma_f32_16x16x32_f16 a[146:149], a[64:67], a[4:7], a[146:149]
	v_mfma_f32_16x16x32_f16 a[150:153], a[72:75], a[4:7], a[150:153]
	v_mfma_f32_16x16x32_f16 a[154:157], a[80:83], a[4:7], a[154:157]
	ds_read_b128 a[0:3], v8 offset:33792
	ds_read_b128 a[4:7], v8 offset:33856
	ds_read_b128 a[8:11], v8 offset:34048
	ds_read_b128 a[12:15], v8 offset:34112
	ds_read_b128 a[16:19], v8 offset:34304
	ds_read_b128 a[20:23], v8 offset:34368
	ds_read_b128 a[24:27], v8 offset:34560
	ds_read_b128 a[28:31], v8 offset:34624
	ds_read_b128 a[32:35], v8 offset:50688
	ds_read_b128 a[36:39], v8 offset:50752
	ds_read_b128 a[40:43], v8 offset:50944
	ds_read_b128 a[44:47], v8 offset:51008
	ds_read_b128 a[48:51], v8 offset:51200
	ds_read_b128 a[52:55], v8 offset:51264
	ds_read_b128 a[56:59], v8 offset:51456
	ds_read_b128 a[60:63], v8 offset:51520
	ds_read_b128 a[64:67], v4
	ds_read_b128 a[68:71], v4 offset:64
	ds_read_b128 a[72:75], v4 offset:256
	ds_read_b128 a[76:79], v4 offset:320
	ds_read_b128 a[80:83], v4 offset:512
	ds_read_b128 a[84:87], v4 offset:576
	ds_read_b128 a[88:91], v4 offset:768
	ds_read_b128 a[92:95], v4 offset:832
	v_accvgpr_read_b32 v4, a196
	v_accvgpr_read_b32 v5, a197
	v_accvgpr_read_b32 v6, a198
	v_accvgpr_read_b32 v7, a199
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[158:161], a[64:67], a[0:3], v[158:161]
	buffer_load_dwordx4 v14, s[0:3], 0 offen lds
	s_mov_b32 m0, s38
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 a[200:203], a[72:75], a[0:3], a[200:203]
	buffer_load_dwordx4 v15, s[0:3], 0 offen lds
	s_mov_b32 m0, s39
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[204:207], a[80:83], a[0:3], a[204:207]
	buffer_load_dwordx4 v16, s[0:3], 0 offen lds
	s_mov_b32 m0, s40
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[138:141], a[88:91], a[0:3], v[138:141]
	buffer_load_dwordx4 v17, s[0:3], 0 offen lds
	s_waitcnt vmcnt(16) lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[142:145], a[64:67], a[8:11], v[142:145]
	s_barrier
	s_mov_b32 m0, s21
	v_mfma_f32_16x16x32_f16 v[146:149], a[72:75], a[8:11], v[146:149]
	v_mfma_f32_16x16x32_f16 v[150:153], a[80:83], a[8:11], v[150:153]
	v_mfma_f32_16x16x32_f16 v[154:157], a[88:91], a[8:11], v[154:157]
	v_mfma_f32_16x16x32_f16 v[162:165], a[64:67], a[16:19], v[162:165]
	v_mfma_f32_16x16x32_f16 v[166:169], a[72:75], a[16:19], v[166:169]
	v_mfma_f32_16x16x32_f16 v[170:173], a[80:83], a[16:19], v[170:173]
	v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[16:19], v[174:177]
	v_mfma_f32_16x16x32_f16 v[178:181], a[64:67], a[24:27], v[178:181]
	v_mfma_f32_16x16x32_f16 v[182:185], a[72:75], a[24:27], v[182:185]
	v_mfma_f32_16x16x32_f16 v[186:189], a[80:83], a[24:27], v[186:189]
	v_mfma_f32_16x16x32_f16 v[190:193], a[88:91], a[24:27], v[190:193]
	v_mfma_f32_16x16x32_f16 v[194:197], a[64:67], a[32:35], v[194:197]
	v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[32:35], v[198:201]
	v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[32:35], v[202:205]
	v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[32:35], v[206:209]
	v_mfma_f32_16x16x32_f16 v[210:213], a[64:67], a[40:43], v[210:213]
	v_mfma_f32_16x16x32_f16 v[214:217], a[72:75], a[40:43], v[214:217]
	v_mfma_f32_16x16x32_f16 v[218:221], a[80:83], a[40:43], v[218:221]
	v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[40:43], v[222:225]
	v_mfma_f32_16x16x32_f16 v[226:229], a[64:67], a[48:51], v[226:229]
	v_mfma_f32_16x16x32_f16 v[230:233], a[72:75], a[48:51], v[230:233]
	v_mfma_f32_16x16x32_f16 v[234:237], a[80:83], a[48:51], v[234:237]
	v_mfma_f32_16x16x32_f16 v[238:241], a[88:91], a[48:51], v[238:241]
	v_mfma_f32_16x16x32_f16 v[242:245], a[64:67], a[56:59], v[242:245]
	v_mfma_f32_16x16x32_f16 v[246:249], a[72:75], a[56:59], v[246:249]
	v_mfma_f32_16x16x32_f16 v[250:253], a[80:83], a[56:59], v[250:253]
	v_mfma_f32_16x16x32_f16 v[4:7], a[88:91], a[56:59], v[4:7]
	v_mfma_f32_16x16x32_f16 v[18:21], a[68:71], a[4:7], v[158:161]
	v_mfma_f32_16x16x32_f16 a[200:203], a[76:79], a[4:7], a[200:203]
	v_mfma_f32_16x16x32_f16 a[204:207], a[84:87], a[4:7], a[204:207]
	v_mfma_f32_16x16x32_f16 v[138:141], a[92:95], a[4:7], v[138:141]
	v_mfma_f32_16x16x32_f16 v[142:145], a[68:71], a[12:15], v[142:145]
	v_mfma_f32_16x16x32_f16 v[146:149], a[76:79], a[12:15], v[146:149]
	v_mfma_f32_16x16x32_f16 v[150:153], a[84:87], a[12:15], v[150:153]
	v_mfma_f32_16x16x32_f16 v[154:157], a[92:95], a[12:15], v[154:157]
	v_mfma_f32_16x16x32_f16 v[162:165], a[68:71], a[20:23], v[162:165]
	v_mfma_f32_16x16x32_f16 v[166:169], a[76:79], a[20:23], v[166:169]
	v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[20:23], v[170:173]
	v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[20:23], v[174:177]
	v_mfma_f32_16x16x32_f16 v[178:181], a[68:71], a[28:31], v[178:181]
	v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[28:31], v[182:185]
	v_mfma_f32_16x16x32_f16 v[186:189], a[84:87], a[28:31], v[186:189]
	v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[28:31], v[190:193]
	v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[36:39], v[194:197]
	v_mfma_f32_16x16x32_f16 v[198:201], a[76:79], a[36:39], v[198:201]
	v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[36:39], v[202:205]
	v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[36:39], v[206:209]
	v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[44:47], v[210:213]
	v_mfma_f32_16x16x32_f16 v[214:217], a[76:79], a[44:47], v[214:217]
	v_mfma_f32_16x16x32_f16 v[218:221], a[84:87], a[44:47], v[218:221]
	v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[44:47], v[222:225]
	v_mfma_f32_16x16x32_f16 v[226:229], a[68:71], a[52:55], v[226:229]
	v_mfma_f32_16x16x32_f16 v[230:233], a[76:79], a[52:55], v[230:233]
	v_mfma_f32_16x16x32_f16 v[234:237], a[84:87], a[52:55], v[234:237]
	v_mfma_f32_16x16x32_f16 v[238:241], a[92:95], a[52:55], v[238:241]
	v_mfma_f32_16x16x32_f16 v[242:245], a[68:71], a[60:63], v[242:245]
	v_mfma_f32_16x16x32_f16 v[246:249], a[76:79], a[60:63], v[246:249]
	v_mfma_f32_16x16x32_f16 v[250:253], a[84:87], a[60:63], v[250:253]
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[60:63], v[4:7]
	ds_read_b128 a[64:67], v26
	ds_read_b128 a[68:71], v26 offset:64
	ds_read_b128 a[72:75], v26 offset:256
	ds_read_b128 a[76:79], v26 offset:320
	ds_read_b128 a[80:83], v26 offset:512
	ds_read_b128 a[84:87], v26 offset:576
	ds_read_b128 a[88:91], v26 offset:768
	ds_read_b128 a[92:95], v26 offset:832
	v_accvgpr_read_b32 v26, a97
	buffer_load_dwordx4 v26, s[16:19], 0 offen lds
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[0:3], v[10:13]
	s_mov_b32 m0, s22
	v_accvgpr_read_b32 v26, a98
	buffer_load_dwordx4 v26, s[16:19], 0 offen lds
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[28:31], a[76:79], a[4:7], v[10:13]
	s_mov_b32 m0, s23
	v_accvgpr_read_b32 v26, a100
	buffer_load_dwordx4 v26, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[8:11], v[22:25]
	s_mov_b32 m0, s41
	v_accvgpr_read_b32 v26, a101
	buffer_load_dwordx4 v26, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[48:51], a[68:71], a[12:15], v[10:13]
	s_mov_b32 m0, s42
	v_accvgpr_read_b32 v26, a102
	buffer_load_dwordx4 v26, s[16:19], 0 offen lds
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[10:13], a[88:91], a[8:11], v[34:37]
	s_mov_b32 m0, s43
	v_accvgpr_read_b32 v26, a103
	buffer_load_dwordx4 v26, s[16:19], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[34:37], a[92:95], a[12:15], v[10:13]
	s_mov_b32 m0, s44
	v_accvgpr_read_b32 v26, a104
	buffer_load_dwordx4 v26, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[16:19], v[38:41]
	s_mov_b32 m0, s45
	v_accvgpr_read_b32 v26, a105
	buffer_load_dwordx4 v26, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[38:41], a[68:71], a[20:23], v[10:13]
	s_and_b32 s17, s24, 0xffff
	s_mov_b32 s16, s20
	s_mov_b32 m0, s46
	v_mfma_f32_16x16x32_f16 v[10:13], a[80:83], a[24:27], v[62:65]
	buffer_load_dwordx4 v9, s[16:19], 0 offen lds
	s_mov_b32 m0, s47
	v_accvgpr_read_b32 v9, a110
	v_mfma_f32_16x16x32_f16 v[62:65], a[84:87], a[28:31], v[10:13]
	buffer_load_dwordx4 v254, s[16:19], 0 offen lds
	s_mov_b32 m0, s48
	s_add_u32 s20, s20, 0x100
	v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[32:35], v[74:77]
	buffer_load_dwordx4 v255, s[16:19], 0 offen lds
	s_mov_b32 m0, s49
	s_addc_u32 s24, s24, 0
	v_mfma_f32_16x16x32_f16 v[74:77], a[76:79], a[36:39], v[10:13]
	buffer_load_dwordx4 v9, s[16:19], 0 offen lds
	s_mov_b32 m0, s50
	s_waitcnt vmcnt(16)
	v_mfma_f32_16x16x32_f16 v[10:13], a[80:83], a[32:35], v[78:81]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_add_u32 s10, s10, 0x100
	v_mfma_f32_16x16x32_f16 v[78:81], a[84:87], a[36:39], v[10:13]
	s_addc_u32 s25, s25, 0
	s_add_i32 s26, s26, 2
	s_cmpk_lt_u32 s26, 0x7c
	v_mfma_f32_16x16x32_f16 v[10:13], a[88:91], a[32:35], v[82:85]
	v_mfma_f32_16x16x32_f16 v[82:85], a[92:95], a[36:39], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[40:43], v[86:89]
	v_mfma_f32_16x16x32_f16 v[86:89], a[68:71], a[44:47], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[40:43], v[90:93]
	v_mfma_f32_16x16x32_f16 v[90:93], a[76:79], a[44:47], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[80:83], a[40:43], v[94:97]
	v_mfma_f32_16x16x32_f16 v[94:97], a[84:87], a[44:47], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[88:91], a[40:43], v[98:101]
	v_mfma_f32_16x16x32_f16 a[166:169], a[64:67], a[0:3], a[166:169]
	v_mfma_f32_16x16x32_f16 a[170:173], a[80:83], a[0:3], a[170:173]
	v_mfma_f32_16x16x32_f16 a[162:165], a[88:91], a[0:3], a[162:165]
	v_accvgpr_mov_b32 a0, a184
	v_accvgpr_mov_b32 a1, a185
	v_accvgpr_mov_b32 a2, a186
	v_mfma_f32_16x16x32_f16 v[98:101], a[92:95], a[44:47], v[10:13]
	v_accvgpr_mov_b32 a3, a187
	v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[48:51], v[102:105]
	v_mfma_f32_16x16x32_f16 a[166:169], a[68:71], a[4:7], a[166:169]
	v_mfma_f32_16x16x32_f16 a[170:173], a[84:87], a[4:7], a[170:173]
	v_mfma_f32_16x16x32_f16 a[162:165], a[92:95], a[4:7], a[162:165]
	v_accvgpr_mov_b32 a4, a188
	v_accvgpr_mov_b32 a5, a189
	v_accvgpr_mov_b32 a6, a190
	v_accvgpr_mov_b32 a7, a191
	v_mfma_f32_16x16x32_f16 v[102:105], a[68:71], a[52:55], v[10:13]
	s_nop 1
	v_accvgpr_read_b32 v44, a170
	v_accvgpr_read_b32 v24, a166
	v_accvgpr_read_b32 v45, a171
	v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[48:51], v[106:109]
	v_accvgpr_read_b32 v46, a172
	v_accvgpr_read_b32 v47, a173
	v_accvgpr_mov_b32 a170, a200
	v_mfma_f32_16x16x32_f16 a[0:3], a[72:75], a[8:11], a[0:3]
	v_accvgpr_read_b32 v25, a167
	v_accvgpr_read_b32 v26, a168
	v_accvgpr_read_b32 v27, a169
	v_mfma_f32_16x16x32_f16 a[4:7], a[80:83], a[8:11], a[4:7]
	v_accvgpr_write_b32 a11, v3
	v_accvgpr_write_b32 a10, v2
	v_accvgpr_write_b32 a9, v1
	v_accvgpr_write_b32 a8, v0
	v_mfma_f32_16x16x32_f16 v[106:109], a[76:79], a[52:55], v[10:13]
	v_accvgpr_read_b32 v0, a161
	v_accvgpr_mov_b32 a171, a201
	v_accvgpr_mov_b32 a172, a202
	v_mfma_f32_16x16x32_f16 a[8:11], a[72:75], a[16:19], a[8:11]
	v_accvgpr_mov_b32 a173, a203
	v_mfma_f32_16x16x32_f16 v[10:13], a[80:83], a[48:51], v[110:113]
	v_mfma_f32_16x16x32_f16 a[0:3], a[76:79], a[12:15], a[0:3]
	v_mfma_f32_16x16x32_f16 a[4:7], a[84:87], a[12:15], a[4:7]
	v_mfma_f32_16x16x32_f16 a[8:11], a[76:79], a[20:23], a[8:11]
	s_nop 5
	v_accvgpr_read_b32 v55, a3
	v_accvgpr_read_b32 v54, a2
	v_accvgpr_read_b32 v53, a1
	v_mfma_f32_16x16x32_f16 a[116:119], a[80:83], a[16:19], a[116:119]
	v_accvgpr_read_b32 v59, a7
	v_accvgpr_read_b32 v52, a0
	v_accvgpr_read_b32 v58, a6
	v_mfma_f32_16x16x32_f16 a[120:123], a[88:91], a[16:19], a[120:123]
	v_accvgpr_read_b32 v71, a11
	v_accvgpr_read_b32 v57, a5
	v_accvgpr_read_b32 v56, a4
	v_mfma_f32_16x16x32_f16 a[124:127], a[64:67], a[24:27], a[124:127]
	v_accvgpr_read_b32 v70, a10
	v_accvgpr_read_b32 v69, a9
	v_accvgpr_read_b32 v68, a8
	v_mfma_f32_16x16x32_f16 a[130:133], a[72:75], a[24:27], a[130:133]
	v_mfma_f32_16x16x32_f16 a[134:137], a[88:91], a[24:27], a[134:137]
	v_mfma_f32_16x16x32_f16 a[138:141], a[64:67], a[32:35], a[138:141]
	v_mfma_f32_16x16x32_f16 v[110:113], a[84:87], a[52:55], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[88:91], a[48:51], v[114:117]
	v_mfma_f32_16x16x32_f16 a[142:145], a[64:67], a[56:59], a[142:145]
	v_mfma_f32_16x16x32_f16 a[146:149], a[72:75], a[56:59], a[146:149]
	v_mfma_f32_16x16x32_f16 a[150:153], a[80:83], a[56:59], a[150:153]
	v_mfma_f32_16x16x32_f16 a[154:157], a[88:91], a[56:59], a[154:157]
	v_mfma_f32_16x16x32_f16 a[116:119], a[84:87], a[20:23], a[116:119]
	v_mfma_f32_16x16x32_f16 a[120:123], a[92:95], a[20:23], a[120:123]
	v_mfma_f32_16x16x32_f16 a[124:127], a[68:71], a[28:31], a[124:127]
	v_mfma_f32_16x16x32_f16 a[130:133], a[76:79], a[28:31], a[130:133]
	v_mfma_f32_16x16x32_f16 a[134:137], a[92:95], a[28:31], a[134:137]
	v_mfma_f32_16x16x32_f16 a[138:141], a[68:71], a[36:39], a[138:141]
	v_mfma_f32_16x16x32_f16 v[114:117], a[92:95], a[52:55], v[10:13]
	v_mfma_f32_16x16x32_f16 a[142:145], a[68:71], a[60:63], a[142:145]
	v_mfma_f32_16x16x32_f16 a[146:149], a[76:79], a[60:63], a[146:149]
	v_mfma_f32_16x16x32_f16 a[150:153], a[84:87], a[60:63], a[150:153]
	v_mfma_f32_16x16x32_f16 a[154:157], a[92:95], a[60:63], a[154:157]
	ds_read_b128 a[56:59], v8
	ds_read_b128 a[60:63], v8 offset:64
	ds_read_b128 a[48:51], v8 offset:256
	ds_read_b128 a[52:55], v8 offset:320
	ds_read_b128 a[40:43], v8 offset:512
	ds_read_b128 a[44:47], v8 offset:576
	ds_read_b128 a[32:35], v8 offset:768
	ds_read_b128 a[36:39], v8 offset:832
	ds_read_b128 a[24:27], v8 offset:16896
	ds_read_b128 a[28:31], v8 offset:16960
	ds_read_b128 a[16:19], v8 offset:17152
	ds_read_b128 a[20:23], v8 offset:17216
	ds_read_b128 a[8:11], v8 offset:17408
	ds_read_b128 a[12:15], v8 offset:17472
	ds_read_b128 a[0:3], v8 offset:17664
	ds_read_b128 a[4:7], v8 offset:17728
	buffer_load_dwordx4 v14, s[16:19], 0 offen lds
	s_mov_b32 m0, s51
	ds_read_b128 a[64:67], v0
	ds_read_b128 a[68:71], v0 offset:64
	ds_read_b128 a[72:75], v0 offset:256
	ds_read_b128 a[76:79], v0 offset:320
	ds_read_b128 a[80:83], v0 offset:512
	ds_read_b128 a[84:87], v0 offset:576
	ds_read_b128 a[88:91], v0 offset:768
	ds_read_b128 a[92:95], v0 offset:832
	buffer_load_dwordx4 v15, s[16:19], 0 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v16, s[16:19], 0 offen lds
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dwordx4 v17, s[16:19], 0 offen lds
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	v_accvgpr_read_b32 v0, a115
	v_lshlrev_b32_e32 v128, 3, v0
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[56:59], v[18:21]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[158:161], a[68:71], a[60:63], v[0:3]
	v_accvgpr_read_b32 v18, a108
	v_accvgpr_write_b32 a197, v41
	v_accvgpr_write_b32 a196, v40
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[56:59], v[138:141]
	v_accvgpr_write_b32 a195, v39
	v_accvgpr_write_b32 a194, v38
	v_accvgpr_write_b32 a113, v105
	v_mfma_f32_16x16x32_f16 v[138:141], a[92:95], a[60:63], v[0:3]
	v_accvgpr_write_b32 a181, v89
	v_accvgpr_write_b32 a112, v104
	v_accvgpr_write_b32 a111, v103
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[48:51], v[142:145]
	v_accvgpr_write_b32 a110, v102
	v_accvgpr_write_b32 a180, v88
	v_accvgpr_write_b32 a179, v87
	v_mfma_f32_16x16x32_f16 v[120:123], a[68:71], a[52:55], v[0:3]
	v_accvgpr_write_b32 a178, v86
	v_accvgpr_write_b32 a169, v77
	v_accvgpr_write_b32 a161, v109
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[48:51], v[146:149]
	v_accvgpr_write_b32 a185, v93
	v_accvgpr_write_b32 a168, v76
	v_accvgpr_write_b32 a167, v75
	v_mfma_f32_16x16x32_f16 v[124:127], a[76:79], a[52:55], v[0:3]
	v_accvgpr_write_b32 a166, v74
	v_accvgpr_write_b32 a160, v108
	v_accvgpr_write_b32 a159, v107
	v_mfma_f32_16x16x32_f16 v[0:3], a[80:83], a[48:51], v[150:153]
	v_accvgpr_write_b32 a158, v106
	v_accvgpr_write_b32 a184, v92
	v_accvgpr_write_b32 a183, v91
	v_mfma_f32_16x16x32_f16 v[130:133], a[84:87], a[52:55], v[0:3]
	v_accvgpr_write_b32 a182, v90
	v_accvgpr_write_b32 a177, v85
	v_accvgpr_write_b32 a189, v97
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[48:51], v[154:157]
	v_accvgpr_write_b32 a193, v101
	s_mul_i32 s0, s9, s12
	v_accvgpr_write_b32 a176, v84
	v_mfma_f32_16x16x32_f16 v[14:17], a[64:67], a[24:27], v[194:197]
	v_accvgpr_write_b32 a175, v83
	v_accvgpr_write_b32 a174, v82
	v_accvgpr_write_b32 a188, v96
	v_mfma_f32_16x16x32_f16 v[134:137], a[92:95], a[52:55], v[0:3]
	v_accvgpr_write_b32 a187, v95
	v_accvgpr_write_b32 a186, v94
	v_accvgpr_write_b32 a192, v100
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[40:43], v[162:165]
	v_accvgpr_write_b32 a191, v99
	v_accvgpr_write_b32 a190, v98
	s_ashr_i32 s1, s0, 31
	v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[28:31], v[14:17]
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s6, s0
	s_addc_u32 s3, s7, s1
	v_mfma_f32_16x16x32_f16 v[14:17], a[72:75], a[24:27], v[198:201]
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[0:1], s[8:9], 1
	v_add_u32_e32 v8, 0x149e0, v18
	v_mfma_f32_16x16x32_f16 v[154:157], a[68:71], a[44:47], v[0:3]
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_lshl_b32 s2, s12, 6
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[40:43], v[166:169]
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s28, s0, s2
	v_mfma_f32_16x16x32_f16 v[198:201], a[76:79], a[28:31], v[14:17]
	s_addc_u32 s15, s1, s3
	s_add_u32 s24, s28, s2
	s_addc_u32 s14, s15, s3
	v_mfma_f32_16x16x32_f16 v[14:17], a[80:83], a[24:27], v[202:205]
	s_add_u32 s20, s24, s2
	s_addc_u32 s11, s14, s3
	s_add_u32 s16, s0, 0x100
	v_mfma_f32_16x16x32_f16 v[162:165], a[76:79], a[44:47], v[0:3]
	s_addc_u32 s10, s1, 0
	v_mfma_f32_16x16x32_f16 v[0:3], a[80:83], a[40:43], v[170:173]
	v_mfma_f32_16x16x32_f16 v[170:173], a[84:87], a[28:31], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[88:91], a[24:27], v[206:209]
	v_mfma_f32_16x16x32_f16 v[166:169], a[84:87], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[40:43], v[174:177]
	v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[28:31], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[64:67], a[16:19], v[210:213]
	v_mfma_f32_16x16x32_f16 v[142:145], a[92:95], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[32:35], v[178:181]
	v_mfma_f32_16x16x32_f16 v[178:181], a[68:71], a[20:23], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[72:75], a[16:19], v[214:217]
	v_mfma_f32_16x16x32_f16 v[146:149], a[68:71], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[32:35], v[182:185]
	v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[20:23], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[80:83], a[16:19], v[218:221]
	v_mfma_f32_16x16x32_f16 v[150:153], a[76:79], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[80:83], a[32:35], v[186:189]
	v_mfma_f32_16x16x32_f16 v[186:189], a[84:87], a[20:23], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[88:91], a[16:19], v[222:225]
	v_mfma_f32_16x16x32_f16 v[10:13], a[88:91], a[32:35], v[190:193]
	v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[20:23], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[64:67], a[8:11], v[226:229]
	v_mfma_f32_16x16x32_f16 v[226:229], a[68:71], a[12:15], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[72:75], a[8:11], v[230:233]
	v_mfma_f32_16x16x32_f16 v[230:233], a[76:79], a[12:15], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[80:83], a[8:11], v[234:237]
	v_mfma_f32_16x16x32_f16 v[202:205], a[84:87], a[12:15], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[88:91], a[8:11], v[238:241]
	v_mfma_f32_16x16x32_f16 v[206:209], a[92:95], a[12:15], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[64:67], a[0:3], v[242:245]
	v_mfma_f32_16x16x32_f16 v[4:7], a[88:91], a[0:3], v[4:7]
	v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[4:7], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[72:75], a[0:3], v[246:249]
	v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[4:7], v[4:7]
	s_nop 4
	v_add_u32_e32 v4, 0x18bc0, v18
	ds_read_b128 v[20:23], v4
	v_mfma_f32_16x16x32_f16 v[214:217], a[76:79], a[4:7], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[80:83], a[0:3], v[250:253]
	s_nop 2
	v_mov_b64_e32 v[252:253], v[64:65]
	v_mov_b64_e32 v[250:251], v[62:63]
	v_mov_b64_e32 v[62:63], v[36:37]
	v_mov_b64_e32 v[60:61], v[34:35]
	ds_read_b128 v[32:35], v4 offset:64
	v_mfma_f32_16x16x32_f16 a[100:103], a[72:75], a[56:59], a[170:173]
	v_mfma_f32_16x16x32_f16 a[100:103], a[76:79], a[60:63], a[100:103]
	ds_read_b128 v[36:39], v4 offset:256
	ds_read_b128 v[40:43], v4 offset:320
	ds_read_b128 a[64:67], v4 offset:512
	ds_read_b128 a[68:71], v4 offset:576
	ds_read_b128 a[72:75], v4 offset:768
	ds_read_b128 a[76:79], v4 offset:832
	v_accvgpr_write_b32 a173, v81
	v_accvgpr_write_b32 a172, v80
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[4:7], v[20:23], a[56:59], v[24:27]
	v_accvgpr_write_b32 a171, v79
	v_accvgpr_write_b32 a170, v78
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[24:27], v[32:35], a[60:63], v[4:7]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[4:7], v[36:39], a[56:59], v[28:31]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[234:237], v[40:43], a[60:63], v[4:7]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], a[56:59], v[44:47]
	v_mfma_f32_16x16x32_f16 v[218:221], a[84:87], a[4:7], v[14:17]
	s_nop 2
	v_accvgpr_read_b32 v14, a162
	v_accvgpr_read_b32 v15, a163
	v_accvgpr_read_b32 v16, a164
	v_accvgpr_read_b32 v17, a165
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[28:31], a[68:71], a[60:63], v[4:7]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], a[56:59], v[14:17]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[44:47], a[76:79], a[60:63], v[4:7]
	s_nop 0
	v_accvgpr_read_b32 v17, a129
	v_mfma_f32_16x16x32_f16 v[4:7], v[20:23], a[48:51], v[48:51]
	v_mfma_f32_16x16x32_f16 v[48:51], v[32:35], a[52:55], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], v[36:39], a[48:51], v[52:55]
	v_mfma_f32_16x16x32_f16 v[52:55], v[40:43], a[52:55], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], a[48:51], v[56:59]
	v_mfma_f32_16x16x32_f16 v[56:59], a[68:71], a[52:55], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], a[48:51], v[60:63]
	v_accvgpr_mov_b32 a48, a116
	v_accvgpr_mov_b32 a49, a117
	v_accvgpr_mov_b32 a50, a118
	v_mfma_f32_16x16x32_f16 v[60:63], a[76:79], a[52:55], v[4:7]
	v_accvgpr_mov_b32 a51, a119
	v_mfma_f32_16x16x32_f16 v[4:7], v[36:39], a[40:43], v[68:71]
	v_mfma_f32_16x16x32_f16 v[68:71], v[40:43], a[44:47], v[4:7]
	v_mfma_f32_16x16x32_f16 a[194:197], v[20:23], a[40:43], a[194:197]
	s_nop 5
	v_accvgpr_read_b32 v4, a120
	v_accvgpr_read_b32 v5, a121
	v_accvgpr_read_b32 v6, a122
	v_accvgpr_read_b32 v7, a123
	v_mfma_f32_16x16x32_f16 a[178:181], v[20:23], a[16:19], a[178:181]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], a[40:43], v[4:7]
	v_mfma_f32_16x16x32_f16 v[238:241], a[76:79], a[44:47], v[4:7]
	v_mfma_f32_16x16x32_f16 a[194:197], v[32:35], a[44:47], a[194:197]
	s_nop 5
	v_accvgpr_read_b32 v4, a124
	v_accvgpr_read_b32 v5, a125
	v_accvgpr_read_b32 v6, a126
	v_accvgpr_read_b32 v7, a127
	v_mfma_f32_16x16x32_f16 a[166:169], v[36:39], a[24:27], a[166:169]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[4:7], v[20:23], a[32:35], v[4:7]
	v_mfma_f32_16x16x32_f16 v[242:245], v[32:35], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 a[178:181], v[32:35], a[20:23], a[178:181]
	s_nop 5
	v_accvgpr_read_b32 v4, a130
	v_accvgpr_read_b32 v5, a131
	v_accvgpr_read_b32 v6, a132
	v_accvgpr_read_b32 v7, a133
	v_mfma_f32_16x16x32_f16 a[130:133], v[20:23], a[8:11], a[110:113]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[4:7], v[36:39], a[32:35], v[4:7]
	v_mfma_f32_16x16x32_f16 v[246:249], v[40:43], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], a[32:35], v[250:253]
	v_mfma_f32_16x16x32_f16 v[250:253], a[68:71], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 a[182:185], v[36:39], a[16:19], a[182:185]
	s_nop 5
	v_accvgpr_read_b32 v4, a134
	v_accvgpr_read_b32 v5, a135
	v_accvgpr_read_b32 v6, a136
	v_accvgpr_read_b32 v7, a137
	v_mfma_f32_16x16x32_f16 a[130:133], v[32:35], a[12:15], a[130:133]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], a[32:35], v[4:7]
	v_accvgpr_mov_b32 a32, a138
	v_accvgpr_mov_b32 a33, a139
	v_accvgpr_mov_b32 a34, a140
	v_mfma_f32_16x16x32_f16 v[64:67], a[76:79], a[36:39], v[4:7]
	v_accvgpr_mov_b32 a35, a141
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[32:35], v[20:23], a[24:27], a[32:35]
	v_accvgpr_read_b32 v4, a142
	v_accvgpr_read_b32 v5, a143
	v_accvgpr_read_b32 v6, a144
	v_accvgpr_read_b32 v7, a145
	v_mfma_f32_16x16x32_f16 a[162:165], v[32:35], a[28:31], a[32:35]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[20:23], v[20:23], a[0:3], v[4:7]
	s_nop 2
	v_accvgpr_read_b32 v4, a146
	v_accvgpr_read_b32 v5, a147
	v_accvgpr_read_b32 v6, a148
	v_accvgpr_read_b32 v7, a149
	v_mfma_f32_16x16x32_f16 a[134:137], v[36:39], a[8:11], a[158:161]
	v_mfma_f32_16x16x32_f16 v[20:23], v[32:35], a[4:7], v[20:23]
	v_mfma_f32_16x16x32_f16 v[32:35], v[36:39], a[0:3], v[4:7]
	s_nop 2
	v_accvgpr_read_b32 v4, a150
	v_accvgpr_read_b32 v5, a151
	v_accvgpr_read_b32 v6, a152
	v_accvgpr_read_b32 v7, a153
	v_mfma_f32_16x16x32_f16 a[104:107], a[80:83], a[56:59], a[204:207]
	v_accvgpr_write_b32 a209, v117
	v_accvgpr_write_b32 a208, v116
	s_nop 0
	v_accvgpr_write_b32 a205, v113
	v_mfma_f32_16x16x32_f16 v[36:39], a[64:67], a[0:3], v[4:7]
	v_accvgpr_write_b32 a204, v112
	v_accvgpr_write_b32 a203, v111
	v_accvgpr_write_b32 a202, v110
	v_accvgpr_read_b32 v4, a154
	v_accvgpr_write_b32 a207, v115
	v_accvgpr_write_b32 a206, v114
	v_accvgpr_read_b32 v5, a155
	v_accvgpr_read_b32 v6, a156
	v_accvgpr_read_b32 v7, a157
	v_mfma_f32_16x16x32_f16 a[48:51], a[64:67], a[40:43], a[48:51]
	v_mfma_f32_16x16x32_f16 a[166:169], v[40:43], a[28:31], a[166:169]
	v_mfma_f32_16x16x32_f16 a[170:173], a[64:67], a[24:27], a[170:173]
	v_mfma_f32_16x16x32_f16 a[174:177], a[72:75], a[24:27], a[174:177]
	v_mfma_f32_16x16x32_f16 a[182:185], v[40:43], a[20:23], a[182:185]
	v_mfma_f32_16x16x32_f16 a[186:189], a[64:67], a[16:19], a[186:189]
	v_mfma_f32_16x16x32_f16 a[190:193], a[72:75], a[16:19], a[190:193]
	v_mfma_f32_16x16x32_f16 a[134:137], v[40:43], a[12:15], a[134:137]
	v_mfma_f32_16x16x32_f16 a[138:141], a[64:67], a[8:11], a[202:205]
	v_mfma_f32_16x16x32_f16 a[158:161], a[72:75], a[8:11], a[206:209]
	v_mfma_f32_16x16x32_f16 v[32:35], v[40:43], a[4:7], v[32:35]
	v_mfma_f32_16x16x32_f16 v[40:43], a[72:75], a[0:3], v[4:7]
	s_nop 2
	v_accvgpr_read_b32 v4, a99
	v_mfma_f32_16x16x32_f16 a[104:107], a[84:87], a[60:63], a[104:107]
	v_add_u32_e32 v5, 0x1cdc0, v18
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[10:13], a[92:95], a[36:39], v[10:13]
	v_mfma_f32_16x16x32_f16 a[198:201], a[68:71], a[44:47], a[48:51]
	v_mfma_f32_16x16x32_f16 a[170:173], a[68:71], a[28:31], a[170:173]
	v_mfma_f32_16x16x32_f16 a[174:177], a[76:79], a[28:31], a[174:177]
	v_mfma_f32_16x16x32_f16 a[186:189], a[68:71], a[20:23], a[186:189]
	v_mfma_f32_16x16x32_f16 a[190:193], a[76:79], a[20:23], a[190:193]
	v_mfma_f32_16x16x32_f16 a[138:141], a[68:71], a[12:15], a[138:141]
	v_mfma_f32_16x16x32_f16 a[158:161], a[76:79], a[12:15], a[158:161]
	v_mfma_f32_16x16x32_f16 v[36:39], a[68:71], a[4:7], v[36:39]
	v_mfma_f32_16x16x32_f16 v[40:43], a[76:79], a[4:7], v[40:43]
	ds_read_b128 a[0:3], v4 offset:33792
	ds_read_b128 a[4:7], v4 offset:33856
	ds_read_b128 a[8:11], v4 offset:34048
	ds_read_b128 a[12:15], v4 offset:34112
	ds_read_b128 a[16:19], v4 offset:34304
	ds_read_b128 a[20:23], v4 offset:34368
	ds_read_b128 a[24:27], v4 offset:34560
	ds_read_b128 a[28:31], v4 offset:34624
	ds_read_b128 a[32:35], v4 offset:50688
	ds_read_b128 a[36:39], v4 offset:50752
	ds_read_b128 a[40:43], v4 offset:50944
	ds_read_b128 a[44:47], v4 offset:51008
	ds_read_b128 a[48:51], v4 offset:51200
	ds_read_b128 a[52:55], v4 offset:51264
	ds_read_b128 a[56:59], v4 offset:51456
	ds_read_b128 a[60:63], v4 offset:51520
	v_accvgpr_read_b32 v4, a96
	ds_read_b128 a[64:67], v8
	ds_read_b128 a[68:71], v8 offset:64
	ds_read_b128 a[72:75], v8 offset:256
	ds_read_b128 a[76:79], v8 offset:320
	ds_read_b128 a[80:83], v8 offset:512
	ds_read_b128 a[84:87], v8 offset:576
	ds_read_b128 a[88:91], v8 offset:768
	ds_read_b128 a[92:95], v8 offset:832
	v_lshrrev_b32_e32 v8, 4, v4
	v_or_b32_e32 v9, 16, v8
	v_or_b32_e32 v14, 32, v8
	v_or_b32_e32 v15, 48, v8
	v_mul_lo_u32 v129, v8, s12
	v_mul_lo_u32 v254, v9, s12
	v_accvgpr_read_b32 v6, a100
	v_accvgpr_read_b32 v7, a101
	v_accvgpr_read_b32 v8, a102
	v_accvgpr_read_b32 v9, a103
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[72:75], a[64:67], a[0:3], v[158:161]
	v_mul_lo_u32 v255, v14, s12
	v_mul_lo_u32 v16, v15, s12
	s_add_u32 s12, s28, 0x100
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[76:79], a[72:75], a[0:3], v[6:9]
	s_addc_u32 s9, s15, 0
	s_add_u32 s8, s24, 0x100
	s_addc_u32 s7, s14, 0
	v_accvgpr_read_b32 v6, a104
	v_accvgpr_read_b32 v7, a105
	v_accvgpr_read_b32 v8, a106
	v_accvgpr_read_b32 v9, a107
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[84:87], a[88:91], a[0:3], v[138:141]
	s_add_u32 s4, s20, 0x100
	s_addc_u32 s6, s11, 0
	ds_read_b128 a[96:99], v5
	ds_read_b128 a[100:103], v5 offset:64
	ds_read_b128 a[104:107], v5 offset:256
	ds_read_b128 a[108:111], v5 offset:320
	ds_read_b128 a[112:115], v5 offset:512
	ds_read_b128 a[116:119], v5 offset:576
	ds_read_b128 a[120:123], v5 offset:768
	ds_read_b128 a[124:127], v5 offset:832
	v_mfma_f32_16x16x32_f16 v[80:83], a[80:83], a[0:3], v[6:9]
	s_lshr_b32 s2, s5, 2
	s_movk_i32 s5, 0x2e00
	s_and_b32 s3, s13, 0x80
	v_mfma_f32_16x16x32_f16 v[72:75], a[68:71], a[4:7], v[72:75]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_and_b32_e32 v6, 0x70, v17
	v_mfma_f32_16x16x32_f16 v[76:79], a[76:79], a[4:7], v[76:79]
	s_and_b32 s1, s1, 0xffff
	s_and_b32 s29, s15, 0xffff
	s_and_b32 s25, s14, 0xffff
	v_mfma_f32_16x16x32_f16 v[80:83], a[84:87], a[4:7], v[80:83]
	s_and_b32 s21, s11, 0xffff
	s_and_b32 s17, s10, 0xffff
	s_and_b32 s13, s9, 0xffff
	v_mfma_f32_16x16x32_f16 v[88:91], a[64:67], a[8:11], v[120:123]
	s_and_b32 s9, s7, 0xffff
	v_mfma_f32_16x16x32_f16 v[84:87], a[92:95], a[4:7], v[84:87]
	v_mfma_f32_16x16x32_f16 v[92:95], a[72:75], a[8:11], v[124:127]
	v_mfma_f32_16x16x32_f16 v[8:11], a[88:91], a[24:27], v[10:13]
	s_nop 2
	v_cvt_pk_f16_f32 v12, v72, v73
	v_cvt_pk_f16_f32 v72, v76, v77
	v_cvt_pk_f16_f32 v76, v80, v81
	v_accvgpr_read_b32 v81, a128
	v_mfma_f32_16x16x32_f16 v[88:91], a[68:71], a[12:15], v[88:91]
	v_and_b32_e32 v7, 1, v81
	v_cvt_pk_f16_f32 v77, v82, v83
	v_cvt_pk_f16_f32 v83, v86, v87
	v_mfma_f32_16x16x32_f16 v[96:99], a[80:83], a[8:11], v[130:133]
	v_lshlrev_b32_e32 v5, 8, v81
	v_lshlrev_b32_e32 v80, 12, v7
	v_and_b32_e32 v86, 16, v81
	v_mfma_f32_16x16x32_f16 v[92:95], a[76:79], a[12:15], v[92:95]
	v_lshlrev_b32_e32 v81, 4, v86
	v_and_or_b32 v5, v5, s5, v80
	v_mov_b32_e32 v80, 0x70
	v_mfma_f32_16x16x32_f16 v[100:103], a[88:91], a[8:11], v[134:137]
	v_or3_b32 v5, s3, v81, v5
	v_bitop3_b32 v87, s2, v17, v80 bitop3:0x78
	v_cvt_pk_f16_f32 v14, v88, v89
	v_mfma_f32_16x16x32_f16 v[96:99], a[84:87], a[12:15], v[96:99]
	v_or_b32_e32 v88, v5, v87
	s_movk_i32 s2, 0x60
	v_cvt_pk_f16_f32 v13, v74, v75
	v_mfma_f32_16x16x32_f16 v[100:103], a[92:95], a[12:15], v[100:103]
	v_cvt_pk_f16_f32 v73, v78, v79
	v_cvt_pk_f16_f32 v15, v90, v91
	v_cvt_pk_f16_f32 v74, v92, v93
	v_cvt_pk_f16_f32 v75, v94, v95
	v_add_u32_e32 v80, 0, v88
	v_xad_u32 v81, v88, 32, 0
	v_bitop3_b32 v5, v5, s2, v87 bitop3:0x36
	ds_write_b128 v80, v[12:15]
	ds_write_b128 v81, v[72:75]
	v_add_u32_e32 v75, 0, v5
	v_and_b32_e32 v5, 0xe0, v4
	v_lshlrev_b32_e32 v12, 4, v5
	v_lshrrev_b32_e32 v5, 1, v5
	v_cvt_pk_f16_f32 v78, v96, v97
	v_cvt_pk_f16_f32 v79, v98, v99
	v_xad_u32 v74, v88, 64, 0
	v_lshlrev_b32_e32 v13, 8, v86
	v_bitop3_b32 v5, v12, v5, v6 bitop3:0x36
	v_lshl_add_u32 v6, v7, 13, 0
	v_cvt_pk_f16_f32 v82, v84, v85
	v_cvt_pk_f16_f32 v84, v100, v101
	v_cvt_pk_f16_f32 v85, v102, v103
	ds_write_b128 v74, v[76:79]
	v_add3_u32 v76, v6, v13, v5
	v_mfma_f32_16x16x32_f16 v[104:107], a[64:67], a[16:19], v[154:157]
	ds_write_b128 v75, v[82:85]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[108:111], a[72:75], a[16:19], v[162:165]
	ds_read_b128 v[82:85], v76
	ds_read_b128 v[88:91], v76 offset:256
	ds_read_b128 v[92:95], v76 offset:128
	ds_read_b128 v[98:101], v76 offset:384
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_mfma_f32_16x16x32_f16 v[112:115], a[80:83], a[16:19], v[166:169]
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v86, v82
	v_mov_b32_e32 v87, v83
	v_add_lshl_u32 v78, v129, v128, 1
	v_mfma_f32_16x16x32_f16 v[120:123], a[64:67], a[24:27], v[146:149]
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v96, v92
	v_mov_b32_e32 v97, v93
	v_add_lshl_u32 v77, v254, v128, 1
	v_mfma_f32_16x16x32_f16 v[124:127], a[72:75], a[24:27], v[150:153]
	buffer_store_dwordx4 v[86:89], v78, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[96:99], v77, s[0:3], 0 offen
	v_add_lshl_u32 v73, v255, v128, 1
	v_mfma_f32_16x16x32_f16 v[116:119], a[88:91], a[16:19], v[142:145]
	v_mov_b32_e32 v86, v90
	v_mov_b32_e32 v87, v91
	v_mov_b32_e32 v96, v100
	v_mfma_f32_16x16x32_f16 v[0:3], a[80:83], a[24:27], v[0:3]
	v_mov_b32_e32 v97, v101
	v_add_lshl_u32 v72, v16, v128, 1
	buffer_store_dwordx4 v[84:87], v73, s[0:3], 0 offen
	v_mfma_f32_16x16x32_f16 v[104:107], a[68:71], a[20:23], v[104:107]
	buffer_store_dwordx4 v[94:97], v72, s[0:3], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[108:111], a[76:79], a[20:23], v[108:111]
	s_mov_b32 s30, s2
	s_mov_b32 s31, s3
	s_mov_b32 s26, s2
	v_mfma_f32_16x16x32_f16 v[112:115], a[84:87], a[20:23], v[112:115]
	v_cvt_pk_f16_f32 v102, v104, v105
	v_cvt_pk_f16_f32 v103, v106, v107
	s_mov_b32 s27, s3
	v_mfma_f32_16x16x32_f16 v[120:123], a[68:71], a[28:31], v[120:123]
	v_cvt_pk_f16_f32 v106, v108, v109
	v_cvt_pk_f16_f32 v107, v110, v111
	s_mov_b32 s22, s2
	v_mfma_f32_16x16x32_f16 v[124:127], a[76:79], a[28:31], v[124:127]
	v_cvt_pk_f16_f32 v110, v112, v113
	v_cvt_pk_f16_f32 v111, v114, v115
	s_mov_b32 s23, s3
	v_mfma_f32_16x16x32_f16 v[116:119], a[92:95], a[20:23], v[116:119]
	v_cvt_pk_f16_f32 v104, v120, v121
	v_cvt_pk_f16_f32 v105, v122, v123
	s_mov_b32 s18, s2
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[28:31], v[0:3]
	v_cvt_pk_f16_f32 v108, v124, v125
	v_cvt_pk_f16_f32 v109, v126, v127
	s_mov_b32 s19, s3
	v_mfma_f32_16x16x32_f16 v[8:11], a[92:95], a[28:31], v[8:11]
	v_cvt_pk_f16_f32 v114, v116, v117
	v_cvt_pk_f16_f32 v115, v118, v119
	s_mov_b32 s14, s2
	v_mfma_f32_16x16x32_f16 v[12:15], a[64:67], a[32:35], v[194:197]
	v_cvt_pk_f16_f32 v112, v0, v1
	v_cvt_pk_f16_f32 v113, v2, v3
	s_mov_b32 s15, s3
	v_mfma_f32_16x16x32_f16 v[82:85], a[72:75], a[32:35], v[198:201]
	v_cvt_pk_f16_f32 v116, v8, v9
	v_cvt_pk_f16_f32 v117, v10, v11
	ds_write_b128 v80, v[102:105]
	ds_write_b128 v81, v[106:109]
	ds_write_b128 v74, v[110:113]
	ds_write_b128 v75, v[114:117]
	v_mfma_f32_16x16x32_f16 v[86:89], a[80:83], a[32:35], v[170:173]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[0:3], v76
	ds_read_b128 v[6:9], v76 offset:256
	ds_read_b128 v[102:105], v76 offset:128
	ds_read_b128 v[108:111], v76 offset:384
	v_mfma_f32_16x16x32_f16 v[90:93], a[88:91], a[32:35], v[174:177]
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v1
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v106, v102
	v_mfma_f32_16x16x32_f16 v[94:97], a[64:67], a[40:43], v[178:181]
	v_mov_b32_e32 v107, v103
	buffer_store_dwordx4 v[4:7], v78, s[28:31], 0 offen
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[106:109], v77, s[28:31], 0 offen
	v_mfma_f32_16x16x32_f16 v[98:101], a[72:75], a[40:43], v[182:185]
	v_mov_b32_e32 v4, v8
	v_mov_b32_e32 v5, v9
	v_mov_b32_e32 v106, v110
	v_mfma_f32_16x16x32_f16 v[128:131], a[80:83], a[40:43], v[186:189]
	v_mov_b32_e32 v107, v111
	buffer_store_dwordx4 v[2:5], v73, s[28:31], 0 offen
	buffer_store_dwordx4 v[104:107], v72, s[28:31], 0 offen
	v_mfma_f32_16x16x32_f16 v[132:135], a[88:91], a[40:43], v[190:193]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s10, s2
	v_mfma_f32_16x16x32_f16 v[12:15], a[68:71], a[36:39], v[12:15]
	s_mov_b32 s11, s3
	s_and_b32 s5, s6, 0xffff
	s_mov_b32 s6, s2
	v_mfma_f32_16x16x32_f16 v[82:85], a[76:79], a[36:39], v[82:85]
	s_mov_b32 s7, s3
	v_mfma_f32_16x16x32_f16 v[86:89], a[84:87], a[36:39], v[86:89]
	s_nop 1
	v_cvt_pk_f16_f32 v12, v12, v13
	v_cvt_pk_f16_f32 v13, v14, v15
	v_mfma_f32_16x16x32_f16 v[90:93], a[92:95], a[36:39], v[90:93]
	s_nop 0
	v_cvt_pk_f16_f32 v82, v82, v83
	v_cvt_pk_f16_f32 v83, v84, v85
	v_mfma_f32_16x16x32_f16 v[94:97], a[68:71], a[44:47], v[94:97]
	v_cvt_pk_f16_f32 v86, v86, v87
	v_cvt_pk_f16_f32 v87, v88, v89
	v_mfma_f32_16x16x32_f16 v[98:101], a[76:79], a[44:47], v[98:101]
	s_nop 0
	v_cvt_pk_f16_f32 v90, v90, v91
	v_cvt_pk_f16_f32 v91, v92, v93
	v_mfma_f32_16x16x32_f16 v[128:131], a[84:87], a[44:47], v[128:131]
	s_nop 0
	v_cvt_pk_f16_f32 v14, v94, v95
	v_cvt_pk_f16_f32 v15, v96, v97
	v_mfma_f32_16x16x32_f16 v[132:135], a[92:95], a[44:47], v[132:135]
	v_cvt_pk_f16_f32 v84, v98, v99
	v_cvt_pk_f16_f32 v85, v100, v101
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[48:51], v[226:229]
	s_nop 0
	v_cvt_pk_f16_f32 v88, v128, v129
	v_cvt_pk_f16_f32 v89, v130, v131
	v_mfma_f32_16x16x32_f16 v[8:11], a[72:75], a[48:51], v[230:233]
	s_nop 0
	v_cvt_pk_f16_f32 v92, v132, v133
	v_cvt_pk_f16_f32 v93, v134, v135
	ds_write_b128 v80, v[12:15]
	ds_write_b128 v81, v[82:85]
	ds_write_b128 v74, v[86:89]
	ds_write_b128 v75, v[90:93]
	v_mfma_f32_16x16x32_f16 v[102:105], a[80:83], a[48:51], v[202:205]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[82:85], v76
	ds_read_b128 v[88:91], v76 offset:256
	ds_read_b128 v[92:95], v76 offset:128
	ds_read_b128 v[98:101], v76 offset:384
	v_mfma_f32_16x16x32_f16 v[106:109], a[88:91], a[48:51], v[206:209]
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v86, v82
	v_mov_b32_e32 v87, v83
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v96, v92
	v_mfma_f32_16x16x32_f16 v[110:113], a[64:67], a[56:59], v[210:213]
	v_mov_b32_e32 v97, v93
	buffer_store_dwordx4 v[86:89], v78, s[24:27], 0 offen
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[96:99], v77, s[24:27], 0 offen
	v_mfma_f32_16x16x32_f16 v[118:121], a[80:83], a[56:59], v[218:221]
	v_mov_b32_e32 v86, v90
	v_mov_b32_e32 v87, v91
	v_mov_b32_e32 v96, v100
	v_mfma_f32_16x16x32_f16 v[122:125], a[88:91], a[56:59], v[222:225]
	v_mov_b32_e32 v97, v101
	buffer_store_dwordx4 v[84:87], v73, s[24:27], 0 offen
	buffer_store_dwordx4 v[94:97], v72, s[24:27], 0 offen
	v_mfma_f32_16x16x32_f16 v[114:117], a[72:75], a[56:59], v[214:217]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[8:11], a[76:79], a[52:55], v[8:11]
	v_mfma_f32_16x16x32_f16 v[102:105], a[84:87], a[52:55], v[102:105]
	s_nop 5
	v_cvt_pk_f16_f32 v0, v0, v1
	v_cvt_pk_f16_f32 v1, v2, v3
	v_mfma_f32_16x16x32_f16 v[106:109], a[92:95], a[52:55], v[106:109]
	v_cvt_pk_f16_f32 v8, v8, v9
	v_cvt_pk_f16_f32 v9, v10, v11
	v_mfma_f32_16x16x32_f16 v[110:113], a[68:71], a[60:63], v[110:113]
	v_cvt_pk_f16_f32 v102, v102, v103
	v_cvt_pk_f16_f32 v103, v104, v105
	v_mfma_f32_16x16x32_f16 v[118:121], a[84:87], a[60:63], v[118:121]
	s_nop 1
	v_cvt_pk_f16_f32 v106, v106, v107
	v_cvt_pk_f16_f32 v107, v108, v109
	v_mfma_f32_16x16x32_f16 v[122:125], a[92:95], a[60:63], v[122:125]
	v_cvt_pk_f16_f32 v2, v110, v111
	v_cvt_pk_f16_f32 v3, v112, v113
	v_mfma_f32_16x16x32_f16 v[114:117], a[76:79], a[60:63], v[114:117]
	v_cvt_pk_f16_f32 v104, v118, v119
	v_cvt_pk_f16_f32 v105, v120, v121
	v_mfma_f32_16x16x32_f16 v[12:15], a[96:99], a[0:3], v[24:27]
	s_nop 1
	v_cvt_pk_f16_f32 v108, v122, v123
	v_cvt_pk_f16_f32 v109, v124, v125
	v_mfma_f32_16x16x32_f16 v[94:97], a[96:99], a[8:11], v[48:51]
	v_cvt_pk_f16_f32 v10, v114, v115
	v_cvt_pk_f16_f32 v11, v116, v117
	ds_write_b128 v80, v[0:3]
	ds_write_b128 v81, v[8:11]
	ds_write_b128 v74, v[102:105]
	ds_write_b128 v75, v[106:109]
	v_mfma_f32_16x16x32_f16 v[82:85], a[104:107], a[0:3], v[234:237]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[0:3], v76
	ds_read_b128 v[6:9], v76 offset:256
	ds_read_b128 v[102:105], v76 offset:128
	ds_read_b128 v[108:111], v76 offset:384
	v_mfma_f32_16x16x32_f16 v[86:89], a[112:115], a[0:3], v[28:31]
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v1
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v106, v102
	v_mfma_f32_16x16x32_f16 v[90:93], a[120:123], a[0:3], v[44:47]
	v_mov_b32_e32 v107, v103
	buffer_store_dwordx4 v[4:7], v78, s[20:23], 0 offen
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[106:109], v77, s[20:23], 0 offen
	v_mfma_f32_16x16x32_f16 v[98:101], a[104:107], a[8:11], v[52:55]
	v_mov_b32_e32 v4, v8
	v_mov_b32_e32 v5, v9
	v_mov_b32_e32 v106, v110
	v_mfma_f32_16x16x32_f16 v[126:129], a[112:115], a[8:11], v[56:59]
	v_mov_b32_e32 v107, v111
	buffer_store_dwordx4 v[2:5], v73, s[20:23], 0 offen
	buffer_store_dwordx4 v[104:107], v72, s[20:23], 0 offen
	v_mfma_f32_16x16x32_f16 v[130:133], a[120:123], a[8:11], v[60:63]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_accvgpr_read_b32 v0, a194
	v_mfma_f32_16x16x32_f16 v[12:15], a[100:103], a[4:7], v[12:15]
	v_accvgpr_read_b32 v4, a198
	v_accvgpr_read_b32 v1, a195
	v_accvgpr_read_b32 v2, a196
	v_mfma_f32_16x16x32_f16 v[94:97], a[100:103], a[12:15], v[94:97]
	v_accvgpr_read_b32 v3, a197
	v_accvgpr_read_b32 v5, a199
	v_accvgpr_read_b32 v6, a200
	v_mfma_f32_16x16x32_f16 v[82:85], a[108:111], a[4:7], v[82:85]
	v_cvt_pk_f16_f32 v12, v12, v13
	v_cvt_pk_f16_f32 v13, v14, v15
	v_accvgpr_read_b32 v7, a201
	v_mfma_f32_16x16x32_f16 v[86:89], a[116:119], a[4:7], v[86:89]
	v_cvt_pk_f16_f32 v14, v94, v95
	v_cvt_pk_f16_f32 v15, v96, v97
	v_mfma_f32_16x16x32_f16 v[90:93], a[124:127], a[4:7], v[90:93]
	s_nop 0
	v_cvt_pk_f16_f32 v82, v82, v83
	v_cvt_pk_f16_f32 v83, v84, v85
	v_mfma_f32_16x16x32_f16 v[98:101], a[108:111], a[12:15], v[98:101]
	s_nop 0
	v_cvt_pk_f16_f32 v86, v86, v87
	v_cvt_pk_f16_f32 v87, v88, v89
	v_mfma_f32_16x16x32_f16 v[126:129], a[116:119], a[12:15], v[126:129]
	v_cvt_pk_f16_f32 v90, v90, v91
	v_cvt_pk_f16_f32 v91, v92, v93
	v_mfma_f32_16x16x32_f16 v[130:133], a[124:127], a[12:15], v[130:133]
	s_nop 0
	v_cvt_pk_f16_f32 v84, v98, v99
	v_cvt_pk_f16_f32 v85, v100, v101
	v_mfma_f32_16x16x32_f16 v[0:3], a[96:99], a[16:19], v[0:3]
	s_nop 0
	v_cvt_pk_f16_f32 v88, v126, v127
	v_cvt_pk_f16_f32 v89, v128, v129
	v_mfma_f32_16x16x32_f16 v[8:11], a[104:107], a[16:19], v[68:71]
	v_cvt_pk_f16_f32 v92, v130, v131
	v_cvt_pk_f16_f32 v93, v132, v133
	ds_write_b128 v80, v[12:15]
	ds_write_b128 v81, v[82:85]
	ds_write_b128 v74, v[86:89]
	ds_write_b128 v75, v[90:93]
	v_accvgpr_read_b32 v12, a162
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[82:85], v76
	ds_read_b128 v[88:91], v76 offset:256
	ds_read_b128 v[92:95], v76 offset:128
	ds_read_b128 v[98:101], v76 offset:384
	v_accvgpr_read_b32 v13, a163
	v_accvgpr_read_b32 v14, a164
	v_accvgpr_read_b32 v15, a165
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v86, v82
	v_mov_b32_e32 v87, v83
	v_mfma_f32_16x16x32_f16 v[12:15], a[96:99], a[32:35], v[12:15]
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[86:89], v78, s[16:19], 0 offen
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v96, v92
	v_mov_b32_e32 v97, v93
	v_mov_b32_e32 v86, v90
	v_mov_b32_e32 v87, v91
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[96:99], v77, s[16:19], 0 offen
	buffer_store_dwordx4 v[84:87], v73, s[16:19], 0 offen
	v_mfma_f32_16x16x32_f16 v[102:105], a[112:115], a[16:19], v[4:7]
	v_mov_b32_e32 v96, v100
	v_mov_b32_e32 v97, v101
	buffer_store_dwordx4 v[94:97], v72, s[16:19], 0 offen
	v_mfma_f32_16x16x32_f16 v[82:85], a[100:103], a[36:39], v[12:15]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_nop 0
	v_accvgpr_read_b32 v12, a166
	v_accvgpr_read_b32 v13, a167
	v_accvgpr_read_b32 v14, a168
	v_accvgpr_read_b32 v15, a169
	v_mfma_f32_16x16x32_f16 v[110:113], a[96:99], a[24:27], v[242:245]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[12:15], a[104:107], a[32:35], v[12:15]
	v_mfma_f32_16x16x32_f16 v[86:89], a[108:111], a[36:39], v[12:15]
	v_mfma_f32_16x16x32_f16 v[114:117], a[104:107], a[24:27], v[246:249]
	s_nop 5
	v_accvgpr_read_b32 v12, a170
	v_accvgpr_read_b32 v13, a171
	v_accvgpr_read_b32 v14, a172
	v_accvgpr_read_b32 v15, a173
	v_mfma_f32_16x16x32_f16 v[4:7], a[120:123], a[24:27], v[64:67]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[12:15], a[112:115], a[32:35], v[12:15]
	v_mfma_f32_16x16x32_f16 v[44:47], a[116:119], a[36:39], v[12:15]
	v_mfma_f32_16x16x32_f16 v[106:109], a[120:123], a[16:19], v[238:241]
	s_nop 5
	v_accvgpr_read_b32 v12, a174
	v_accvgpr_read_b32 v13, a175
	v_accvgpr_read_b32 v14, a176
	v_accvgpr_read_b32 v15, a177
	v_mfma_f32_16x16x32_f16 v[118:121], a[112:115], a[24:27], v[250:253]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[12:15], a[120:123], a[32:35], v[12:15]
	v_mfma_f32_16x16x32_f16 v[48:51], a[124:127], a[36:39], v[12:15]
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[20:23], v[0:3]
	s_nop 5
	v_accvgpr_read_b32 v12, a178
	v_accvgpr_read_b32 v13, a179
	v_accvgpr_read_b32 v14, a180
	v_accvgpr_read_b32 v15, a181
	v_mfma_f32_16x16x32_f16 v[8:11], a[108:111], a[20:23], v[8:11]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[12:15], a[96:99], a[40:43], v[12:15]
	v_cvt_pk_f16_f32 v0, v0, v1
	v_cvt_pk_f16_f32 v1, v2, v3
	v_mfma_f32_16x16x32_f16 v[52:55], a[100:103], a[44:47], v[12:15]
	s_nop 2
	v_cvt_pk_f16_f32 v8, v8, v9
	v_cvt_pk_f16_f32 v9, v10, v11
	v_mfma_f32_16x16x32_f16 v[102:105], a[116:119], a[20:23], v[102:105]
	v_accvgpr_read_b32 v12, a182
	v_accvgpr_read_b32 v13, a183
	v_accvgpr_read_b32 v14, a184
	v_accvgpr_read_b32 v15, a185
	v_mfma_f32_16x16x32_f16 v[110:113], a[100:103], a[28:31], v[110:113]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[12:15], a[104:107], a[40:43], v[12:15]
	v_mfma_f32_16x16x32_f16 v[56:59], a[108:111], a[44:47], v[12:15]
	s_nop 4
	v_cvt_pk_f16_f32 v2, v110, v111
	v_cvt_pk_f16_f32 v3, v112, v113
	v_mfma_f32_16x16x32_f16 v[114:117], a[108:111], a[28:31], v[114:117]
	v_accvgpr_read_b32 v12, a186
	v_accvgpr_read_b32 v13, a187
	v_accvgpr_read_b32 v14, a188
	v_accvgpr_read_b32 v15, a189
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[28:31], v[4:7]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[12:15], a[112:115], a[40:43], v[12:15]
	s_nop 0
	v_cvt_pk_f16_f32 v10, v114, v115
	v_cvt_pk_f16_f32 v11, v116, v117
	v_mfma_f32_16x16x32_f16 v[60:63], a[116:119], a[44:47], v[12:15]
	s_nop 1
	v_cvt_pk_f16_f32 v26, v4, v5
	v_cvt_pk_f16_f32 v27, v6, v7
	v_mfma_f32_16x16x32_f16 v[106:109], a[124:127], a[20:23], v[106:109]
	v_accvgpr_read_b32 v12, a190
	v_accvgpr_read_b32 v13, a191
	v_accvgpr_read_b32 v14, a192
	v_accvgpr_read_b32 v15, a193
	v_mfma_f32_16x16x32_f16 v[118:121], a[116:119], a[28:31], v[118:121]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[12:15], a[120:123], a[40:43], v[12:15]
	s_nop 0
	v_cvt_pk_f16_f32 v24, v106, v107
	v_cvt_pk_f16_f32 v25, v108, v109
	v_mfma_f32_16x16x32_f16 v[64:67], a[124:127], a[44:47], v[12:15]
	s_nop 3
	v_cvt_pk_f16_f32 v12, v102, v103
	v_cvt_pk_f16_f32 v13, v104, v105
	v_cvt_pk_f16_f32 v14, v118, v119
	v_cvt_pk_f16_f32 v15, v120, v121
	ds_write_b128 v80, v[0:3]
	ds_write_b128 v81, v[8:11]
	ds_write_b128 v74, v[12:15]
	ds_write_b128 v75, v[24:27]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[0:3], v76
	ds_read_b128 v[6:9], v76 offset:256
	ds_read_b128 v[10:13], v76 offset:128
	ds_read_b128 v[26:29], v76 offset:384
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v1
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[4:7], v78, s[12:15], 0 offen
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v24, v10
	v_mov_b32_e32 v25, v11
	v_mov_b32_e32 v4, v8
	v_mov_b32_e32 v5, v9
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[24:27], v77, s[12:15], 0 offen
	buffer_store_dwordx4 v[2:5], v73, s[12:15], 0 offen
	v_accvgpr_read_b32 v0, a130
	v_accvgpr_read_b32 v1, a131
	v_accvgpr_read_b32 v2, a132
	v_accvgpr_read_b32 v3, a133
	v_mov_b32_e32 v14, v28
	v_mov_b32_e32 v15, v29
	v_mfma_f32_16x16x32_f16 v[0:3], a[96:99], a[48:51], v[0:3]
	buffer_store_dwordx4 v[12:15], v72, s[12:15], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[24:27], a[100:103], a[52:55], v[0:3]
	s_nop 3
	v_accvgpr_read_b32 v0, a134
	v_accvgpr_read_b32 v1, a135
	v_accvgpr_read_b32 v2, a136
	v_accvgpr_read_b32 v3, a137
	v_cvt_pk_f16_f32 v24, v24, v25
	v_cvt_pk_f16_f32 v25, v26, v27
	v_mfma_f32_16x16x32_f16 v[0:3], a[104:107], a[48:51], v[0:3]
	v_mfma_f32_16x16x32_f16 v[28:31], a[108:111], a[52:55], v[0:3]
	s_nop 6
	v_accvgpr_read_b32 v0, a138
	v_accvgpr_read_b32 v1, a139
	v_accvgpr_read_b32 v2, a140
	v_accvgpr_read_b32 v3, a141
	v_cvt_pk_f16_f32 v28, v28, v29
	v_cvt_pk_f16_f32 v29, v30, v31
	v_mfma_f32_16x16x32_f16 v[0:3], a[112:115], a[48:51], v[0:3]
	v_mfma_f32_16x16x32_f16 v[12:15], a[116:119], a[52:55], v[0:3]
	s_nop 6
	v_accvgpr_read_b32 v0, a158
	v_accvgpr_read_b32 v1, a159
	v_accvgpr_read_b32 v2, a160
	v_accvgpr_read_b32 v3, a161
	v_cvt_pk_f16_f32 v12, v12, v13
	v_cvt_pk_f16_f32 v13, v14, v15
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[48:51], v[0:3]
	v_mfma_f32_16x16x32_f16 v[16:19], a[124:127], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[96:99], a[56:59], v[20:23]
	v_mfma_f32_16x16x32_f16 v[20:23], a[100:103], a[60:63], v[0:3]
	s_nop 5
	v_cvt_pk_f16_f32 v16, v16, v17
	v_cvt_pk_f16_f32 v17, v18, v19
	v_mfma_f32_16x16x32_f16 v[0:3], a[104:107], a[56:59], v[32:35]
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[60:63], v[0:3]
	s_nop 1
	v_cvt_pk_f16_f32 v32, v82, v83
	v_cvt_pk_f16_f32 v33, v84, v85
	v_cvt_pk_f16_f32 v34, v52, v53
	v_mfma_f32_16x16x32_f16 v[0:3], a[112:115], a[56:59], v[36:39]
	v_cvt_pk_f16_f32 v35, v54, v55
	v_cvt_pk_f16_f32 v26, v20, v21
	v_cvt_pk_f16_f32 v27, v22, v23
	v_mfma_f32_16x16x32_f16 v[8:11], a[116:119], a[60:63], v[0:3]
	v_cvt_pk_f16_f32 v38, v56, v57
	v_cvt_pk_f16_f32 v39, v58, v59
	v_cvt_pk_f16_f32 v36, v86, v87
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[56:59], v[40:43]
	v_cvt_pk_f16_f32 v37, v88, v89
	v_cvt_pk_f16_f32 v30, v4, v5
	v_cvt_pk_f16_f32 v31, v6, v7
	v_cvt_pk_f16_f32 v40, v44, v45
	v_cvt_pk_f16_f32 v41, v46, v47
	v_cvt_pk_f16_f32 v44, v48, v49
	v_cvt_pk_f16_f32 v45, v50, v51
	v_cvt_pk_f16_f32 v42, v60, v61
	v_cvt_pk_f16_f32 v43, v62, v63
	v_cvt_pk_f16_f32 v46, v64, v65
	v_cvt_pk_f16_f32 v47, v66, v67
	ds_write_b128 v80, v[32:35]
	ds_write_b128 v81, v[36:39]
	ds_write_b128 v74, v[40:43]
	ds_write_b128 v75, v[44:47]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[42:45], v76
	ds_read_b128 v[48:51], v76 offset:256
	ds_read_b128 v[32:35], v76 offset:128
	ds_read_b128 v[38:41], v76 offset:384
	v_mfma_f32_16x16x32_f16 v[0:3], a[124:127], a[60:63], v[0:3]
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v46, v42
	v_mov_b32_e32 v47, v43
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v36, v32
	v_mov_b32_e32 v37, v33
	buffer_store_dwordx4 v[46:49], v78, s[8:11], 0 offen
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[36:39], v77, s[8:11], 0 offen
	v_cvt_pk_f16_f32 v14, v8, v9
	v_mov_b32_e32 v46, v50
	v_mov_b32_e32 v47, v51
	v_mov_b32_e32 v36, v40
	v_mov_b32_e32 v37, v41
	v_cvt_pk_f16_f32 v18, v0, v1
	v_cvt_pk_f16_f32 v19, v2, v3
	buffer_store_dwordx4 v[44:47], v73, s[8:11], 0 offen
	buffer_store_dwordx4 v[34:37], v72, s[8:11], 0 offen
	v_cvt_pk_f16_f32 v15, v10, v11
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v80, v[24:27]
	ds_write_b128 v81, v[28:31]
	ds_write_b128 v74, v[12:15]
	ds_write_b128 v75, v[16:19]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[0:3], v76
	ds_read_b128 v[6:9], v76 offset:256
	ds_read_b128 v[10:13], v76 offset:128
	ds_read_b128 v[16:19], v76 offset:384
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v1
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v14, v10
	v_mov_b32_e32 v15, v11
	buffer_store_dwordx4 v[4:7], v78, s[4:7], 0 offen
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[14:17], v77, s[4:7], 0 offen
	v_mov_b32_e32 v4, v8
	v_mov_b32_e32 v5, v9
	v_mov_b32_e32 v14, v18
	v_mov_b32_e32 v15, v19
	buffer_store_dwordx4 v[2:5], v73, s[4:7], 0 offen
	buffer_store_dwordx4 v[12:15], v72, s[4:7], 0 offen
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
		.amdhsa_next_free_vgpr 466
		.amdhsa_next_free_sgpr 55
		.amdhsa_accum_offset 256
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
	.set v8_beyond_hotloop.num_vgpr, 256
	.set v8_beyond_hotloop.num_agpr, 210
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
; codeLenInByte = 14192
; TotalNumSgprs: 61
; NumVgprs: 256
; NumAgprs: 210
; TotalNumVgprs: 466
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 58
; NumSGPRsForWavesPerEU: 61
; NumVGPRsForWavesPerEU: 466
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
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
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
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
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
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
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x65 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x3f DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x2d DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	118                             ; DW_AT_call_line
	.byte	71                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4d:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	19                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x59:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	20                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	0
	.quad	0
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
  - .agpr_count:     210
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
    .vgpr_count:     466
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
