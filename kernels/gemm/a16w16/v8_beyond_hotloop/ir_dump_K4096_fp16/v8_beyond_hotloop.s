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
; %bb.6:
	.file	1 "/var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v8_beyond_hotloop" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.7:
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
	v_mov_b32_e32 v36, v0
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
	v_readfirstlane_b32 s52, v36
	s_xor_b32 s1, s8, s1
	s_sub_i32 s15, s15, s17
	s_bfe_u32 s24, s52, 0x20006
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
	v_lshlrev_b32_e32 v23, 3, v36
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
	s_sub_i32 s15, s9, s8
	s_mul_i32 s0, s15, s0
	s_sub_i32 s54, s1, s0
	s_add_i32 s54, s54, s14
	s_lshl_b32 s1, s54, 8
	s_mul_i32 s8, s1, s11
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[8:9], s[8:9], 1
	s_add_u32 s16, s2, s8
	s_addc_u32 s14, s3, s9
	s_lshl_b32 s0, s15, 8
	v_lshlrev_b32_e32 v0, 1, v36
	s_mul_i32 s8, s0, s12
	v_and_b32_e32 v0, 0x70, v0
	s_ashr_i32 s9, s8, 31
	v_or_b32_e32 v1, s24, v0
	s_lshl_b64 s[8:9], s[8:9], 1
	v_or_b32_e32 v4, 4, v1
	v_or_b32_e32 v6, 8, v1
	v_or_b32_e32 v8, 12, v1
	v_or_b32_e32 v2, 0x80, v1
	v_or_b32_e32 v3, 0x84, v1
	v_or_b32_e32 v5, 0x88, v1
	v_or_b32_e32 v7, 0x8c, v1
	v_and_b32_e32 v0, 56, v23
	s_add_u32 s20, s4, s8
	s_addc_u32 s9, s5, s9
	v_mul_lo_u32 v11, v4, s11
	v_mul_lo_u32 v12, v6, s11
	v_mul_lo_u32 v13, v8, s11
	v_mul_lo_u32 v14, v2, s11
	v_mul_lo_u32 v15, v3, s11
	v_mul_lo_u32 v16, v5, s11
	v_mul_lo_u32 v17, v7, s11
	v_mad_u64_u32 v[2:3], s[4:5], v1, s12, v[0:1]
	v_mad_u64_u32 v[4:5], s[4:5], v4, s12, v[0:1]
	v_mad_u64_u32 v[6:7], s[4:5], v6, s12, v[0:1]
	v_mad_u64_u32 v[8:9], s[4:5], v8, s12, v[0:1]
	s_lshl_b32 s4, s12, 8
	s_add_i32 s55, s10, 63
	s_ashr_i32 s5, s4, 1
	s_ashr_i32 s4, s55, 31
	s_lshr_b32 s4, s4, 26
	s_add_i32 s4, s55, s4
	s_mul_i32 s15, s24, 0x420
	v_mul_lo_u32 v10, v1, s11
	s_ashr_i32 s53, s4, 6
	s_add_i32 s4, s15, 0
	s_and_b32 s17, s14, 0xffff
	s_mov_b32 s18, 0x7ffffffe
	v_add_lshl_u32 v28, v10, v0, 1
	s_mov_b32 m0, s4
	s_add_i32 s8, s4, 0x1080
	buffer_load_dwordx4 v28, s[16:19], 0 offen lds
	v_add_lshl_u32 v29, v11, v0, 1
	s_mov_b32 m0, s8
	s_add_i32 s10, s4, 0x2100
	buffer_load_dwordx4 v29, s[16:19], 0 offen lds
	v_add_lshl_u32 v11, v12, v0, 1
	s_mov_b32 m0, s10
	s_add_i32 s12, s4, 0x3180
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	v_add_lshl_u32 v12, v13, v0, 1
	s_mov_b32 m0, s12
	s_add_i32 s25, s4, 0x4200
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	v_add_lshl_u32 v254, v14, v0, 1
	s_mov_b32 m0, s25
	s_add_i32 s26, s4, 0x5280
	buffer_load_dwordx4 v254, s[16:19], 0 offen lds
	v_add_lshl_u32 v255, v15, v0, 1
	s_mov_b32 m0, s26
	s_add_i32 s27, s4, 0x6300
	buffer_load_dwordx4 v255, s[16:19], 0 offen lds
	v_add_lshl_u32 v7, v16, v0, 1
	s_mov_b32 m0, s27
	s_add_i32 s28, s4, 0x7380
	s_add_i32 s57, 0, 0x107e0
	s_add_i32 s50, s15, 0x1080
	s_add_i32 s51, s15, 0x2100
	s_add_i32 s56, s15, 0x3180
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	v_add_lshl_u32 v9, v17, v0, 1
	s_mov_b32 m0, s28
	s_add_i32 s29, s57, s15
	buffer_load_dwordx4 v9, s[16:19], 0 offen lds
	s_and_b32 s21, s9, 0xffff
	s_mov_b32 s22, s18
	v_lshlrev_b32_e32 v0, 1, v2
	s_mov_b32 m0, s29
	s_add_i32 s30, s57, s50
	s_add_i32 s31, s57, s51
	s_add_i32 s33, s57, s56
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v1, 1, v4
	s_mov_b32 m0, s30
	s_add_u32 s16, s16, 0x80
	buffer_load_dwordx4 v1, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v3, 1, v6
	s_mov_b32 m0, s31
	s_addc_u32 s14, s14, 0
	s_add_i32 s37, 0, 0x18bc0
	buffer_load_dwordx4 v3, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v5, 1, v8
	s_mov_b32 m0, s33
	s_add_i32 s34, s37, s15
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	v_add_lshl_u32 v10, v2, s5, 1
	s_mov_b32 m0, s34
	s_add_i32 s35, s37, s50
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	v_add_lshl_u32 v4, v4, s5, 1
	s_mov_b32 m0, s35
	s_add_i32 s36, s37, s51
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	v_add_lshl_u32 v6, v6, s5, 1
	s_mov_b32 m0, s36
	s_add_i32 s37, s37, s56
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	v_add_lshl_u32 v8, v8, s5, 1
	s_mov_b32 m0, s37
	s_add_i32 s38, s34, 0xfffef840
	buffer_load_dwordx4 v8, s[20:23], 0 offen lds
	s_and_b32 s17, s14, 0xffff
	s_mov_b32 m0, s38
	s_add_i32 s39, s34, 0xffff08c0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v28, s[16:19], 0 offen lds
	s_mov_b32 m0, s39
	s_add_i32 s40, s34, 0xffff1940
	buffer_load_dwordx4 v29, s[16:19], 0 offen lds
	s_mov_b32 m0, s40
	s_add_i32 s41, s34, 0xffff29c0
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s42, s34, 0xffff3a40
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	s_mov_b32 m0, s42
	s_add_i32 s43, s34, 0xffff4ac0
	buffer_load_dwordx4 v254, s[16:19], 0 offen lds
	s_mov_b32 m0, s43
	s_add_i32 s44, s34, 0xffff5b40
	buffer_load_dwordx4 v255, s[16:19], 0 offen lds
	s_mov_b32 m0, s44
	s_add_i32 s45, s34, 0xffff6bc0
	s_add_i32 s49, 0, 0x149e0
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s46, s49, s15
	v_accvgpr_write_b32 a106, v11
	buffer_load_dwordx4 v9, s[16:19], 0 offen lds
	v_add_u32_e32 v11, 0x80, v0
	s_mov_b32 m0, s46
	s_add_i32 s47, s49, s50
	v_mov_b32_e32 v145, v12
	buffer_load_dwordx4 v11, s[20:23], 0 offen lds
	v_add_u32_e32 v12, 0x80, v1
	s_mov_b32 m0, s47
	s_add_i32 s48, s49, s51
	buffer_load_dwordx4 v12, s[20:23], 0 offen lds
	v_add_u32_e32 v13, 0x80, v3
	s_mov_b32 m0, s48
	s_add_i32 s49, s49, s56
	buffer_load_dwordx4 v13, s[20:23], 0 offen lds
	v_add_u32_e32 v5, 0x80, v5
	s_mov_b32 m0, s49
	v_and_b32_e32 v37, 15, v36
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_u32 s20, s20, 0x80
	s_addc_u32 s21, s9, 0
	s_add_i32 s5, 0, 0x1cdc0
	s_add_i32 s22, s5, s15
	s_and_b32 s17, s21, 0xffff
	s_mov_b32 s16, s20
	s_mov_b32 m0, s22
	s_add_i32 s23, s5, s50
	buffer_load_dwordx4 v10, s[16:19], 0 offen lds
	s_mov_b32 m0, s23
	s_add_i32 s50, s5, s51
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_mov_b32 m0, s50
	s_add_i32 s51, s5, s56
	buffer_load_dwordx4 v6, s[16:19], 0 offen lds
	s_mov_b32 m0, s51
	s_cmp_lt_u32 s24, 2
	buffer_load_dwordx4 v8, s[16:19], 0 offen lds
	s_cselect_b64 s[14:15], -1, 0
	s_and_b64 s[16:17], s[14:15], exec
	v_lshlrev_b32_e32 v0, 10, v37
	v_and_b32_e32 v1, 48, v36
	s_cselect_b32 s5, 0, 0x100
	v_or3_b32 v2, v1, s5, v0
	v_lshlrev_b32_e32 v3, 5, v37
	v_add_u32_e32 v2, v2, v3
	v_add_u32_e32 v2, 0, v2
	s_waitcnt vmcnt(20) lgkmcnt(0)
	s_barrier
	ds_read_b128 a[60:63], v2
	ds_read_b128 a[56:59], v2 offset:64
	ds_read_b128 a[48:51], v2 offset:128
	ds_read_b128 a[52:55], v2 offset:192
	ds_read_b128 a[44:47], v2 offset:512
	ds_read_b128 a[40:43], v2 offset:576
	ds_read_b128 a[36:39], v2 offset:640
	ds_read_b128 a[32:35], v2 offset:704
	ds_read_b128 a[28:31], v2 offset:16896
	ds_read_b128 a[24:27], v2 offset:16960
	ds_read_b128 a[20:23], v2 offset:17024
	ds_read_b128 a[16:19], v2 offset:17088
	ds_read_b128 a[12:15], v2 offset:17408
	ds_read_b128 a[8:11], v2 offset:17472
	ds_read_b128 a[4:7], v2 offset:17536
	ds_read_b128 a[0:3], v2 offset:17600
	v_or_b32_e32 v0, v0, v1
	s_and_b32 s5, s52, 64
	v_add_u32_e32 v0, v0, v3
	v_lshl_add_u32 v22, s5, 2, v0
	v_add_u32_e32 v0, s57, v22
	ds_read_b128 a[92:95], v0
	ds_read_b128 a[88:91], v0 offset:64
	ds_read_b128 a[84:87], v0 offset:128
	ds_read_b128 a[80:83], v0 offset:192
	ds_read_b128 a[76:79], v0 offset:512
	ds_read_b128 a[72:75], v0 offset:576
	ds_read_b128 a[68:71], v0 offset:640
	ds_read_b128 a[64:67], v0 offset:704
	s_add_i32 s9, s53, -1
	s_cmpk_lt_i32 s55, 0x80
	s_mov_b32 s52, 0
	s_cbranch_scc1 .LBB0_4
; %bb.1:                                ; %.lr.ph
	s_mul_i32 s11, s11, s54
	s_lshl_b32 s16, s11, 8
	s_ashr_i32 s17, s16, 31
	s_add_i32 s53, s53, -2
	s_lshl_b64 s[16:17], s[16:17], 1
	s_add_u32 s2, s2, s16
	v_mov_b32_e32 v126, 0
	s_addc_u32 s3, s3, s17
	v_mov_b32_e32 v132, v126
	v_mov_b32_e32 v133, v126
	s_add_u32 s2, s2, 0x180
	v_add_u32_e32 v0, 0, v22
	v_mov_b32_e32 v130, v126
	v_mov_b32_e32 v131, v126
	v_mov_b64_e32 v[148:149], v[132:133]
	v_accvgpr_write_b32 a227, v133
	v_accvgpr_write_b32 a235, v133
	v_accvgpr_write_b32 a231, v133
	v_mov_b32_e32 v140, v37
	v_mov_b32_e32 v141, v36
	v_mov_b32_e32 v142, v23
	s_addc_u32 s3, s3, 0
	v_mov_b32_e32 v143, v22
	v_add_u32_e32 v150, 0x18bc0, v0
	v_add_u32_e32 v151, 0x149e0, v0
	v_add_u32_e32 v152, 0x1cdc0, v0
	v_add_u32_e32 v153, 0x107e0, v0
	v_mov_b32_e32 v127, v126
	v_mov_b32_e32 v128, v126
	v_mov_b32_e32 v129, v126
	v_mov_b32_e32 v122, v126
	v_mov_b32_e32 v123, v126
	v_mov_b32_e32 v124, v126
	v_mov_b32_e32 v125, v126
	v_mov_b32_e32 v118, v126
	v_mov_b32_e32 v119, v126
	v_mov_b32_e32 v120, v126
	v_mov_b32_e32 v121, v126
	v_mov_b32_e32 v114, v126
	v_mov_b32_e32 v115, v126
	v_mov_b32_e32 v116, v126
	v_mov_b32_e32 v117, v126
	v_mov_b32_e32 v110, v126
	v_mov_b32_e32 v111, v126
	v_mov_b32_e32 v112, v126
	v_mov_b32_e32 v113, v126
	v_mov_b32_e32 v106, v126
	v_mov_b32_e32 v107, v126
	v_mov_b32_e32 v108, v126
	v_mov_b32_e32 v109, v126
	v_mov_b32_e32 v102, v126
	v_mov_b32_e32 v103, v126
	v_mov_b32_e32 v104, v126
	v_mov_b32_e32 v105, v126
	v_mov_b32_e32 v98, v126
	v_mov_b32_e32 v99, v126
	v_mov_b32_e32 v100, v126
	v_mov_b32_e32 v101, v126
	v_mov_b32_e32 v94, v126
	v_mov_b32_e32 v95, v126
	v_mov_b32_e32 v96, v126
	v_mov_b32_e32 v97, v126
	v_mov_b32_e32 v90, v126
	v_mov_b32_e32 v91, v126
	v_mov_b32_e32 v92, v126
	v_mov_b32_e32 v93, v126
	v_mov_b32_e32 v86, v126
	v_mov_b32_e32 v87, v126
	v_mov_b32_e32 v88, v126
	v_mov_b32_e32 v89, v126
	v_mov_b32_e32 v82, v126
	v_mov_b32_e32 v83, v126
	v_mov_b32_e32 v84, v126
	v_mov_b32_e32 v85, v126
	v_mov_b32_e32 v78, v126
	v_mov_b32_e32 v79, v126
	v_mov_b32_e32 v80, v126
	v_mov_b32_e32 v81, v126
	v_mov_b32_e32 v74, v126
	v_mov_b32_e32 v75, v126
	v_mov_b32_e32 v76, v126
	v_mov_b32_e32 v77, v126
	v_mov_b32_e32 v70, v126
	v_mov_b32_e32 v71, v126
	v_mov_b32_e32 v72, v126
	v_mov_b32_e32 v73, v126
	v_mov_b32_e32 v66, v126
	v_mov_b32_e32 v67, v126
	v_mov_b32_e32 v68, v126
	v_mov_b32_e32 v69, v126
	v_mov_b32_e32 v62, v126
	v_mov_b32_e32 v63, v126
	v_mov_b32_e32 v64, v126
	v_mov_b32_e32 v65, v126
	v_mov_b32_e32 v58, v126
	v_mov_b32_e32 v59, v126
	v_mov_b32_e32 v60, v126
	v_mov_b32_e32 v61, v126
	v_mov_b32_e32 v54, v126
	v_mov_b32_e32 v55, v126
	v_mov_b32_e32 v56, v126
	v_mov_b32_e32 v57, v126
	v_mov_b32_e32 v50, v126
	v_mov_b32_e32 v51, v126
	v_mov_b32_e32 v52, v126
	v_mov_b32_e32 v53, v126
	v_accvgpr_write_b32 a194, v126
	v_accvgpr_write_b32 a195, v126
	v_accvgpr_write_b32 a196, v126
	v_accvgpr_write_b32 a197, v126
	v_mov_b32_e32 v42, v126
	v_mov_b32_e32 v43, v126
	v_mov_b32_e32 v44, v126
	v_mov_b32_e32 v45, v126
	v_mov_b32_e32 v38, v126
	v_mov_b32_e32 v39, v126
	v_mov_b32_e32 v40, v126
	v_mov_b32_e32 v41, v126
	v_mov_b32_e32 v46, v126
	v_mov_b32_e32 v47, v126
	v_mov_b32_e32 v48, v126
	v_mov_b32_e32 v49, v126
	v_mov_b32_e32 v136, v126
	v_mov_b32_e32 v137, v126
	v_mov_b32_e32 v138, v126
	v_mov_b32_e32 v139, v126
	v_accvgpr_write_b32 a120, v126
	v_accvgpr_write_b32 a121, v126
	v_accvgpr_write_b32 a122, v126
	v_accvgpr_write_b32 a123, v126
	v_accvgpr_write_b32 a116, v126
	v_accvgpr_write_b32 a117, v126
	v_accvgpr_write_b32 a118, v126
	v_accvgpr_write_b32 a119, v126
	v_mov_b32_e32 v24, v126
	v_mov_b32_e32 v25, v126
	v_mov_b32_e32 v26, v126
	v_mov_b32_e32 v27, v126
	v_mov_b32_e32 v18, v126
	v_mov_b32_e32 v19, v126
	v_mov_b32_e32 v20, v126
	v_mov_b32_e32 v21, v126
	v_mov_b32_e32 v14, v126
	v_mov_b32_e32 v15, v126
	v_mov_b32_e32 v16, v126
	v_mov_b32_e32 v17, v126
	v_mov_b32_e32 v154, v10
	v_accvgpr_write_b32 a96, v126
	v_mov_b32_e32 v155, v11
	v_accvgpr_write_b32 a97, v126
	v_mov_b32_e32 v156, v12
	v_accvgpr_write_b32 a98, v126
	v_mov_b32_e32 v157, v13
	v_accvgpr_write_b32 a99, v126
	v_mov_b32_e32 v0, v126
	v_mov_b32_e32 v1, v126
	v_mov_b32_e32 v158, v2
	v_mov_b32_e32 v2, v126
	v_mov_b32_e32 v3, v126
	v_mov_b32_e32 v159, v6
	v_mov_b32_e32 v6, v126
	v_mov_b32_e32 v160, v7
	v_mov_b32_e32 v7, v126
	v_mov_b32_e32 v161, v8
	v_mov_b32_e32 v8, v126
	v_mov_b32_e32 v162, v9
	v_mov_b32_e32 v9, v126
	v_mov_b64_e32 v[146:147], v[130:131]
	v_accvgpr_write_b32 a226, v132
	v_accvgpr_write_b32 a225, v131
	v_accvgpr_write_b32 a224, v130
	v_accvgpr_write_b32 a234, v132
	v_accvgpr_write_b32 a233, v131
	v_accvgpr_write_b32 a232, v130
	v_mov_b32_e32 v244, v126
	v_mov_b32_e32 v245, v126
	v_mov_b32_e32 v246, v126
	v_mov_b32_e32 v247, v126
	v_mov_b32_e32 v248, v126
	v_mov_b32_e32 v249, v126
	v_mov_b32_e32 v250, v126
	v_mov_b32_e32 v251, v126
	v_accvgpr_write_b32 a230, v132
	v_accvgpr_write_b32 a229, v131
	v_accvgpr_write_b32 a228, v130
	v_mov_b32_e32 v236, v126
	v_mov_b32_e32 v237, v126
	v_mov_b32_e32 v238, v126
	v_mov_b32_e32 v239, v126
	v_mov_b32_e32 v232, v126
	v_mov_b32_e32 v233, v126
	v_mov_b32_e32 v234, v126
	v_mov_b32_e32 v235, v126
	v_mov_b32_e32 v228, v126
	v_mov_b32_e32 v229, v126
	v_mov_b32_e32 v230, v126
	v_mov_b32_e32 v231, v126
	v_mov_b32_e32 v224, v126
	v_mov_b32_e32 v225, v126
	v_mov_b32_e32 v226, v126
	v_mov_b32_e32 v227, v126
	v_mov_b32_e32 v220, v126
	v_mov_b32_e32 v221, v126
	v_mov_b32_e32 v222, v126
	v_mov_b32_e32 v223, v126
	v_mov_b32_e32 v216, v126
	v_mov_b32_e32 v217, v126
	v_mov_b32_e32 v218, v126
	v_mov_b32_e32 v219, v126
	v_mov_b32_e32 v212, v126
	v_mov_b32_e32 v213, v126
	v_mov_b32_e32 v214, v126
	v_mov_b32_e32 v215, v126
	v_mov_b32_e32 v208, v126
	v_mov_b32_e32 v209, v126
	v_mov_b32_e32 v210, v126
	v_mov_b32_e32 v211, v126
	v_mov_b32_e32 v204, v126
	v_mov_b32_e32 v205, v126
	v_mov_b32_e32 v206, v126
	v_mov_b32_e32 v207, v126
	v_mov_b32_e32 v200, v126
	v_mov_b32_e32 v201, v126
	v_mov_b32_e32 v202, v126
	v_mov_b32_e32 v203, v126
	v_mov_b32_e32 v196, v126
	v_mov_b32_e32 v197, v126
	v_mov_b32_e32 v198, v126
	v_mov_b32_e32 v199, v126
	v_mov_b32_e32 v192, v126
	v_mov_b32_e32 v193, v126
	v_mov_b32_e32 v194, v126
	v_mov_b32_e32 v195, v126
	v_mov_b32_e32 v188, v126
	v_mov_b32_e32 v189, v126
	v_mov_b32_e32 v190, v126
	v_mov_b32_e32 v191, v126
	v_mov_b32_e32 v184, v126
	v_mov_b32_e32 v185, v126
	v_mov_b32_e32 v186, v126
	v_mov_b32_e32 v187, v126
	v_accvgpr_write_b32 a164, v126
	v_accvgpr_write_b32 a165, v126
	v_accvgpr_write_b32 a166, v126
	v_accvgpr_write_b32 a167, v126
	v_accvgpr_write_b32 a160, v126
	v_accvgpr_write_b32 a161, v126
	v_accvgpr_write_b32 a162, v126
	v_accvgpr_write_b32 a163, v126
	v_accvgpr_write_b32 a252, v126
	v_accvgpr_write_b32 a253, v126
	v_accvgpr_write_b32 a254, v126
	v_accvgpr_write_b32 a255, v126
	v_accvgpr_write_b32 a248, v126
	v_accvgpr_write_b32 a249, v126
	v_accvgpr_write_b32 a250, v126
	v_accvgpr_write_b32 a251, v126
	v_accvgpr_write_b32 a244, v126
	v_accvgpr_write_b32 a245, v126
	v_accvgpr_write_b32 a246, v126
	v_accvgpr_write_b32 a247, v126
	v_accvgpr_write_b32 a240, v126
	v_accvgpr_write_b32 a241, v126
	v_accvgpr_write_b32 a242, v126
	v_accvgpr_write_b32 a243, v126
	v_accvgpr_write_b32 a236, v126
	v_accvgpr_write_b32 a237, v126
	v_accvgpr_write_b32 a238, v126
	v_accvgpr_write_b32 a239, v126
	v_accvgpr_write_b32 a112, v126
	v_accvgpr_write_b32 a113, v126
	v_accvgpr_write_b32 a114, v126
	v_accvgpr_write_b32 a115, v126
	v_accvgpr_write_b32 a108, v126
	v_accvgpr_write_b32 a109, v126
	v_accvgpr_write_b32 a110, v126
	v_accvgpr_write_b32 a111, v126
	v_accvgpr_write_b32 a102, v126
	v_accvgpr_write_b32 a103, v126
	v_accvgpr_write_b32 a104, v126
	v_accvgpr_write_b32 a105, v126
	v_accvgpr_write_b32 a222, v4
	v_accvgpr_write_b32 a223, v5
	v_mov_b32_e32 v144, v28
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	v_accvgpr_mov_b32 a127, a105
	v_accvgpr_mov_b32 a126, a104
	v_accvgpr_mov_b32 a125, a103
	v_accvgpr_mov_b32 a124, a102
	v_accvgpr_mov_b32 a100, a224
	v_accvgpr_mov_b32 a101, a225
	v_accvgpr_mov_b32 a102, a226
	v_accvgpr_mov_b32 a103, a227
	v_accvgpr_mov_b32 a131, a111
	v_accvgpr_mov_b32 a130, a110
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[100:103], a[76:79], a[60:63], a[100:103]
	v_accvgpr_mov_b32 a129, a109
	v_accvgpr_mov_b32 a128, a108
	v_accvgpr_write_b32 a108, v244
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 a[224:227], a[72:75], a[56:59], a[100:103]
	v_accvgpr_write_b32 a109, v245
	v_accvgpr_write_b32 a110, v246
	v_accvgpr_write_b32 a111, v247
	v_mfma_f32_16x16x32_f16 v[6:9], a[92:95], a[60:63], v[6:9]
	v_accvgpr_mov_b32 a100, a232
	v_accvgpr_mov_b32 a101, a233
	v_accvgpr_mov_b32 a102, a234
	v_accvgpr_mov_b32 a103, a235
	v_accvgpr_mov_b32 a145, a115
	v_accvgpr_mov_b32 a146, a236
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 a[100:103], a[68:71], a[60:63], a[100:103]
	v_accvgpr_mov_b32 a190, a240
	v_accvgpr_mov_b32 a186, a244
	v_accvgpr_mov_b32 a182, a248
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 a[232:235], a[64:67], a[56:59], a[100:103]
	v_accvgpr_mov_b32 a178, a252
	v_accvgpr_mov_b32 a177, a163
	v_accvgpr_mov_b32 a173, a167
	v_mfma_f32_16x16x32_f16 a[102:105], a[92:95], a[48:51], a[108:111]
	v_accvgpr_mov_b32 a153, a99
	v_accvgpr_mov_b32 a144, a114
	v_accvgpr_mov_b32 a143, a113
	v_accvgpr_mov_b32 a108, a228
	v_accvgpr_mov_b32 a109, a229
	v_accvgpr_mov_b32 a110, a230
	v_accvgpr_mov_b32 a111, a231
	v_mfma_f32_16x16x32_f16 v[4:7], a[88:91], a[56:59], v[6:9]
	v_accvgpr_mov_b32 a142, a112
	v_accvgpr_mov_b32 a147, a237
	v_accvgpr_mov_b32 a148, a238
	v_mfma_f32_16x16x32_f16 a[108:111], a[76:79], a[48:51], a[108:111]
	v_accvgpr_mov_b32 a149, a239
	v_accvgpr_mov_b32 a191, a241
	v_accvgpr_mov_b32 a192, a242
	v_accvgpr_mov_b32 a193, a243
	v_accvgpr_mov_b32 a187, a245
	v_accvgpr_mov_b32 a188, a246
	v_accvgpr_mov_b32 a189, a247
	v_accvgpr_mov_b32 a183, a249
	v_accvgpr_mov_b32 a184, a250
	v_accvgpr_mov_b32 a185, a251
	v_accvgpr_mov_b32 a179, a253
	v_accvgpr_mov_b32 a180, a254
	v_accvgpr_mov_b32 a181, a255
	v_accvgpr_mov_b32 a176, a162
	v_accvgpr_mov_b32 a175, a161
	v_accvgpr_mov_b32 a174, a160
	v_accvgpr_mov_b32 a172, a166
	v_accvgpr_mov_b32 a171, a165
	v_accvgpr_mov_b32 a170, a164
	v_accvgpr_write_b32 a166, v184
	v_accvgpr_write_b32 a162, v188
	v_accvgpr_write_b32 a158, v192
	v_accvgpr_write_b32 a217, v199
	v_accvgpr_write_b32 a221, v203
	v_accvgpr_write_b32 a239, v207
	v_accvgpr_write_b32 a243, v211
	v_accvgpr_write_b32 a247, v215
	v_accvgpr_write_b32 a251, v219
	v_accvgpr_write_b32 a255, v223
	v_accvgpr_write_b32 a198, v224
	v_accvgpr_write_b32 a202, v228
	v_accvgpr_write_b32 a112, v232
	v_accvgpr_write_b32 a206, v236
	v_accvgpr_mov_b32 a152, a98
	v_accvgpr_mov_b32 a151, a97
	v_accvgpr_mov_b32 a150, a96
	v_accvgpr_write_b32 a213, v133
	v_mov_b64_e32 v[30:31], v[136:137]
	v_accvgpr_write_b32 a96, v146
	v_accvgpr_write_b32 a167, v185
	v_accvgpr_write_b32 a168, v186
	v_accvgpr_write_b32 a169, v187
	v_accvgpr_write_b32 a163, v189
	v_accvgpr_write_b32 a164, v190
	v_accvgpr_write_b32 a165, v191
	v_accvgpr_write_b32 a159, v193
	v_accvgpr_write_b32 a160, v194
	v_accvgpr_write_b32 a161, v195
	v_accvgpr_write_b32 a216, v198
	v_accvgpr_write_b32 a215, v197
	v_accvgpr_write_b32 a214, v196
	v_accvgpr_write_b32 a220, v202
	v_accvgpr_write_b32 a219, v201
	v_accvgpr_write_b32 a218, v200
	v_accvgpr_write_b32 a238, v206
	v_accvgpr_write_b32 a237, v205
	v_accvgpr_write_b32 a236, v204
	v_accvgpr_write_b32 a242, v210
	v_accvgpr_write_b32 a241, v209
	v_accvgpr_write_b32 a240, v208
	v_accvgpr_write_b32 a246, v214
	v_accvgpr_write_b32 a245, v213
	v_accvgpr_write_b32 a244, v212
	v_accvgpr_write_b32 a250, v218
	v_accvgpr_write_b32 a249, v217
	v_accvgpr_write_b32 a248, v216
	v_accvgpr_write_b32 a254, v222
	v_accvgpr_write_b32 a253, v221
	v_accvgpr_write_b32 a252, v220
	v_accvgpr_write_b32 a199, v225
	v_accvgpr_write_b32 a200, v226
	v_accvgpr_write_b32 a201, v227
	v_accvgpr_write_b32 a203, v229
	v_accvgpr_write_b32 a204, v230
	v_accvgpr_write_b32 a205, v231
	v_accvgpr_write_b32 a113, v233
	v_accvgpr_write_b32 a114, v234
	v_accvgpr_write_b32 a115, v235
	v_accvgpr_write_b32 a207, v237
	v_accvgpr_write_b32 a208, v238
	v_accvgpr_write_b32 a209, v239
	v_accvgpr_write_b32 a212, v132
	v_accvgpr_write_b32 a211, v131
	v_accvgpr_write_b32 a210, v130
	v_accvgpr_mov_b32 a137, a123
	v_mov_b64_e32 v[32:33], v[138:139]
	v_accvgpr_mov_b32 a141, a119
	v_mov_b64_e32 v[136:137], v[16:17]
	v_accvgpr_write_b32 a97, v147
	v_accvgpr_write_b32 a98, v148
	v_accvgpr_write_b32 a99, v149
	v_accvgpr_mov_b32 a136, a122
	v_accvgpr_mov_b32 a135, a121
	v_accvgpr_mov_b32 a134, a120
	v_accvgpr_mov_b32 a140, a118
	v_accvgpr_mov_b32 a139, a117
	v_accvgpr_mov_b32 a138, a116
	v_mov_b64_e32 v[134:135], v[14:15]
	v_mfma_f32_16x16x32_f16 a[96:99], a[84:87], a[60:63], a[96:99]
	v_accvgpr_write_b32 a157, v7
	v_accvgpr_write_b32 a156, v6
	v_accvgpr_write_b32 a155, v5
	v_mfma_f32_16x16x32_f16 v[12:15], a[84:87], a[48:51], v[248:251]
	v_accvgpr_write_b32 a154, v4
	v_accvgpr_read_b32 v243, a197
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mfma_f32_16x16x32_f16 a[228:231], a[72:75], a[52:55], a[108:111]
	v_accvgpr_read_b32 v242, a196
	v_accvgpr_read_b32 v241, a195
	v_accvgpr_read_b32 v240, a194
	v_mfma_f32_16x16x32_f16 a[116:119], a[68:71], a[48:51], a[206:209]
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 a[112:115], a[92:95], a[44:47], a[112:115]
	v_mov_b64_e32 v[4:5], v[0:1]
	s_add_u32 s16, s2, 0xffffff80
	s_addc_u32 s11, s3, -1
	v_mfma_f32_16x16x32_f16 a[108:111], a[84:87], a[44:47], a[202:205]
	s_cmp_eq_u32 s53, s52
	s_cselect_b64 vcc, -1, 0
	v_mov_b32_e32 v8, v144
	v_mfma_f32_16x16x32_f16 a[120:123], a[76:79], a[44:47], a[198:201]
	v_bfrev_b32_e32 v9, 1
	s_and_b32 s17, s11, 0xffff
	v_cndmask_b32_e32 v28, v8, v9, vcc
	v_mfma_f32_16x16x32_f16 a[252:255], a[68:71], a[44:47], a[252:255]
	s_mov_b32 m0, s4
	v_cndmask_b32_e32 v36, v29, v9, vcc
	buffer_load_dwordx4 v28, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 a[248:251], a[92:95], a[36:39], a[248:251]
	s_mov_b32 m0, s8
	v_accvgpr_read_b32 v138, a106
	buffer_load_dwordx4 v36, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 a[244:247], a[84:87], a[36:39], a[244:247]
	v_cndmask_b32_e32 v37, v138, v9, vcc
	s_mov_b32 m0, s10
	v_mov_b32_e32 v8, v145
	v_mfma_f32_16x16x32_f16 a[240:243], a[76:79], a[36:39], a[240:243]
	buffer_load_dwordx4 v37, s[16:19], 0 offen lds
	v_cndmask_b32_e32 v22, v8, v9, vcc
	s_mov_b32 m0, s12
	v_mfma_f32_16x16x32_f16 a[236:239], a[68:71], a[36:39], a[236:239]
	v_mov_b32_e32 v34, v254
	buffer_load_dwordx4 v22, s[16:19], 0 offen lds
	v_cndmask_b32_e32 v254, v34, v9, vcc
	v_mfma_f32_16x16x32_f16 a[218:221], a[92:95], a[28:31], a[218:221]
	s_mov_b32 m0, s25
	v_mov_b32_e32 v35, v255
	buffer_load_dwordx4 v254, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 a[214:217], a[84:87], a[28:31], a[214:217]
	v_cndmask_b32_e32 v255, v35, v9, vcc
	s_mov_b32 m0, s26
	v_mov_b32_e32 v8, v160
	v_mfma_f32_16x16x32_f16 a[158:161], a[76:79], a[28:31], a[158:161]
	buffer_load_dwordx4 v255, s[16:19], 0 offen lds
	v_cndmask_b32_e32 v16, v8, v9, vcc
	s_mov_b32 m0, s27
	v_mfma_f32_16x16x32_f16 a[162:165], a[68:71], a[28:31], a[162:165]
	v_mov_b32_e32 v8, v162
	buffer_load_dwordx4 v16, s[16:19], 0 offen lds
	v_cndmask_b32_e32 v11, v8, v9, vcc
	v_mfma_f32_16x16x32_f16 a[166:169], a[92:95], a[20:23], a[166:169]
	s_mov_b32 m0, s28
	v_mov_b32_e32 v8, v155
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 a[170:173], a[84:87], a[20:23], a[170:173]
	s_and_b32 s17, s21, 0xffff
	s_mov_b32 s16, s20
	v_accvgpr_write_b32 a100, v29
	v_mfma_f32_16x16x32_f16 a[174:177], a[76:79], a[20:23], a[174:177]
	v_cndmask_b32_e32 v29, v8, v9, vcc
	s_mov_b32 m0, s29
	v_mov_b32_e32 v8, v156
	v_mfma_f32_16x16x32_f16 a[178:181], a[68:71], a[20:23], a[178:181]
	buffer_load_dwordx4 v29, s[16:19], 0 offen lds
	v_cndmask_b32_e32 v17, v8, v9, vcc
	s_mov_b32 m0, s30
	v_mfma_f32_16x16x32_f16 a[182:185], a[92:95], a[12:15], a[182:185]
	v_mov_b32_e32 v8, v157
	buffer_load_dwordx4 v17, s[16:19], 0 offen lds
	v_cndmask_b32_e32 v10, v8, v9, vcc
	v_mfma_f32_16x16x32_f16 a[186:189], a[84:87], a[12:15], a[186:189]
	s_mov_b32 m0, s31
	v_accvgpr_read_b32 v8, a223
	buffer_load_dwordx4 v10, s[16:19], 0 offen lds
	v_mfma_f32_16x16x32_f16 a[190:193], a[76:79], a[12:15], a[190:193]
	v_cndmask_b32_e32 v23, v8, v9, vcc
	s_mov_b32 m0, s33
	s_mov_b32 s56, s2
	v_mfma_f32_16x16x32_f16 a[146:149], a[68:71], a[12:15], a[146:149]
	buffer_load_dwordx4 v23, s[16:19], 0 offen lds
	s_add_u32 s16, s20, 0x80
	s_addc_u32 s11, s21, 0
	v_mfma_f32_16x16x32_f16 a[92:95], a[92:95], a[4:7], a[142:145]
	s_and_b32 s17, s11, 0xffff
	s_mov_b32 m0, s34
	s_and_b32 s57, s3, 0xffff
	v_mfma_f32_16x16x32_f16 a[84:87], a[84:87], a[4:7], a[128:131]
	v_accvgpr_write_b32 a145, v27
	v_accvgpr_write_b32 a144, v26
	v_accvgpr_write_b32 a143, v25
	v_mfma_f32_16x16x32_f16 a[76:79], a[76:79], a[4:7], a[124:127]
	v_accvgpr_write_b32 a142, v24
	s_mov_b32 s58, s18
	s_mov_b32 s59, s19
	v_mfma_f32_16x16x32_f16 a[210:213], a[68:71], a[4:7], a[210:213]
	s_add_u32 s20, s20, 0x100
	s_addc_u32 s21, s21, 0
	s_add_i32 s52, s52, 2
	v_mfma_f32_16x16x32_f16 a[96:99], a[80:83], a[56:59], a[96:99]
	v_mfma_f32_16x16x32_f16 a[102:105], a[88:91], a[52:55], a[102:105]
	v_mfma_f32_16x16x32_f16 v[250:253], a[80:83], a[52:55], v[12:15]
	v_mfma_f32_16x16x32_f16 a[116:119], a[64:67], a[52:55], a[116:119]
	v_mfma_f32_16x16x32_f16 a[112:115], a[88:91], a[40:43], a[112:115]
	v_mfma_f32_16x16x32_f16 a[108:111], a[80:83], a[40:43], a[108:111]
	v_mfma_f32_16x16x32_f16 a[120:123], a[72:75], a[40:43], a[120:123]
	v_mfma_f32_16x16x32_f16 a[252:255], a[64:67], a[40:43], a[252:255]
	v_mfma_f32_16x16x32_f16 a[248:251], a[88:91], a[32:35], a[248:251]
	v_mfma_f32_16x16x32_f16 a[244:247], a[80:83], a[32:35], a[244:247]
	v_mfma_f32_16x16x32_f16 a[240:243], a[72:75], a[32:35], a[240:243]
	v_mfma_f32_16x16x32_f16 a[236:239], a[64:67], a[32:35], a[236:239]
	v_mfma_f32_16x16x32_f16 a[218:221], a[88:91], a[24:27], a[218:221]
	v_mfma_f32_16x16x32_f16 a[214:217], a[80:83], a[24:27], a[214:217]
	v_mfma_f32_16x16x32_f16 a[158:161], a[72:75], a[24:27], a[158:161]
	v_mfma_f32_16x16x32_f16 a[162:165], a[64:67], a[24:27], a[162:165]
	v_mfma_f32_16x16x32_f16 a[166:169], a[88:91], a[16:19], a[166:169]
	v_mfma_f32_16x16x32_f16 a[170:173], a[80:83], a[16:19], a[170:173]
	v_mfma_f32_16x16x32_f16 a[174:177], a[72:75], a[16:19], a[174:177]
	v_mfma_f32_16x16x32_f16 a[178:181], a[64:67], a[16:19], a[178:181]
	v_mfma_f32_16x16x32_f16 a[182:185], a[88:91], a[8:11], a[182:185]
	v_mfma_f32_16x16x32_f16 a[186:189], a[80:83], a[8:11], a[186:189]
	v_mfma_f32_16x16x32_f16 a[190:193], a[72:75], a[8:11], a[190:193]
	v_mfma_f32_16x16x32_f16 a[194:197], a[64:67], a[8:11], a[146:149]
	v_mfma_f32_16x16x32_f16 a[198:201], a[88:91], a[0:3], a[92:95]
	s_nop 1
	v_accvgpr_write_b32 a149, v21
	v_accvgpr_write_b32 a148, v20
	v_accvgpr_write_b32 a147, v19
	v_mfma_f32_16x16x32_f16 a[202:205], a[80:83], a[0:3], a[84:87]
	v_accvgpr_write_b32 a146, v18
	v_accvgpr_read_b32 v18, a102
	v_accvgpr_read_b32 v19, a103
	v_mfma_f32_16x16x32_f16 a[206:209], a[72:75], a[0:3], a[76:79]
	v_accvgpr_read_b32 v20, a104
	v_accvgpr_read_b32 v21, a105
	v_mfma_f32_16x16x32_f16 a[210:213], a[64:67], a[0:3], a[210:213]
	ds_read_b128 v[0:3], v150
	ds_read_b128 a[64:67], v150 offset:64
	ds_read_b128 a[68:71], v150 offset:128
	ds_read_b128 a[72:75], v150 offset:192
	ds_read_b128 a[76:79], v150 offset:512
	ds_read_b128 a[80:83], v150 offset:576
	ds_read_b128 a[84:87], v150 offset:640
	ds_read_b128 a[88:91], v150 offset:704
	s_waitcnt vmcnt(16) lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[12:15], v[0:3], a[60:63], v[126:129]
	s_barrier
	v_mfma_f32_16x16x32_f16 v[126:129], a[64:67], a[56:59], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[68:71], a[60:63], v[122:125]
	v_mfma_f32_16x16x32_f16 v[122:125], a[72:75], a[56:59], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[76:79], a[60:63], v[118:121]
	v_mfma_f32_16x16x32_f16 v[118:121], a[80:83], a[56:59], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[84:87], a[60:63], v[114:117]
	v_mfma_f32_16x16x32_f16 v[114:117], a[88:91], a[56:59], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], v[0:3], a[48:51], v[110:113]
	v_mfma_f32_16x16x32_f16 v[110:113], a[64:67], a[52:55], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[68:71], a[48:51], v[106:109]
	v_mfma_f32_16x16x32_f16 v[106:109], a[72:75], a[52:55], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[76:79], a[48:51], v[102:105]
	v_mfma_f32_16x16x32_f16 v[102:105], a[80:83], a[52:55], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[84:87], a[48:51], v[98:101]
	v_mfma_f32_16x16x32_f16 v[98:101], a[88:91], a[52:55], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], v[0:3], a[44:47], v[94:97]
	v_mfma_f32_16x16x32_f16 v[94:97], a[64:67], a[40:43], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[68:71], a[44:47], v[90:93]
	v_mfma_f32_16x16x32_f16 v[90:93], a[72:75], a[40:43], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[76:79], a[44:47], v[86:89]
	v_mfma_f32_16x16x32_f16 v[86:89], a[80:83], a[40:43], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[84:87], a[44:47], v[82:85]
	v_mfma_f32_16x16x32_f16 v[82:85], a[88:91], a[40:43], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], v[0:3], a[36:39], v[78:81]
	v_mfma_f32_16x16x32_f16 v[78:81], a[64:67], a[32:35], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[68:71], a[36:39], v[74:77]
	v_mfma_f32_16x16x32_f16 v[74:77], a[72:75], a[32:35], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[76:79], a[36:39], v[70:73]
	v_mfma_f32_16x16x32_f16 v[70:73], a[80:83], a[32:35], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[84:87], a[36:39], v[66:69]
	v_mfma_f32_16x16x32_f16 v[66:69], a[88:91], a[32:35], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], v[0:3], a[28:31], v[62:65]
	v_mfma_f32_16x16x32_f16 v[62:65], a[64:67], a[24:27], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[68:71], a[28:31], v[58:61]
	v_mfma_f32_16x16x32_f16 v[58:61], a[72:75], a[24:27], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[76:79], a[28:31], v[54:57]
	v_mfma_f32_16x16x32_f16 v[54:57], a[80:83], a[24:27], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[84:87], a[28:31], v[50:53]
	v_mfma_f32_16x16x32_f16 v[50:53], a[88:91], a[24:27], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], v[0:3], a[20:23], v[240:243]
	v_mfma_f32_16x16x32_f16 v[12:15], a[64:67], a[16:19], v[12:15]
	v_mfma_f32_16x16x32_f16 a[142:145], a[84:87], a[12:15], a[142:145]
	v_mfma_f32_16x16x32_f16 a[146:149], v[0:3], a[4:7], a[146:149]
	s_nop 5
	v_accvgpr_write_b32 a129, v15
	v_accvgpr_write_b32 a128, v14
	v_accvgpr_write_b32 a127, v13
	v_accvgpr_write_b32 a126, v12
	v_mfma_f32_16x16x32_f16 v[12:15], a[68:71], a[20:23], v[42:45]
	v_mfma_f32_16x16x32_f16 v[42:45], a[72:75], a[16:19], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[76:79], a[20:23], v[38:41]
	v_mfma_f32_16x16x32_f16 v[38:41], a[80:83], a[16:19], v[12:15]
	v_mfma_f32_16x16x32_f16 v[12:15], a[84:87], a[20:23], v[46:49]
	v_mfma_f32_16x16x32_f16 v[12:15], a[88:91], a[16:19], v[12:15]
	v_mfma_f32_16x16x32_f16 a[16:19], a[68:71], a[12:15], a[134:137]
	v_mfma_f32_16x16x32_f16 a[134:137], a[72:75], a[8:11], a[16:19]
	s_nop 5
	v_accvgpr_write_b32 a133, v15
	v_accvgpr_write_b32 a132, v14
	v_accvgpr_write_b32 a131, v13
	v_accvgpr_write_b32 a130, v12
	v_mfma_f32_16x16x32_f16 v[12:15], v[0:3], a[12:15], v[30:33]
	v_mfma_f32_16x16x32_f16 v[30:33], a[64:67], a[8:11], v[12:15]
	v_mfma_f32_16x16x32_f16 a[16:19], a[76:79], a[12:15], a[138:141]
	v_mfma_f32_16x16x32_f16 v[12:15], a[68:71], a[4:7], v[134:137]
	v_mfma_f32_16x16x32_f16 a[150:153], a[76:79], a[4:7], a[150:153]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 a[138:141], a[80:83], a[8:11], a[16:19]
	s_nop 1
	v_mov_b32_e32 v4, v154
	v_cndmask_b32_e32 v4, v4, v9, vcc
	v_accvgpr_write_b32 a107, v4
	v_mfma_f32_16x16x32_f16 a[142:145], a[88:91], a[8:11], a[142:145]
	v_mov_b32_e32 v5, v158
	v_mfma_f32_16x16x32_f16 a[146:149], a[64:67], a[0:3], a[146:149]
	v_mfma_f32_16x16x32_f16 v[12:15], a[72:75], a[0:3], v[12:15]
	v_mfma_f32_16x16x32_f16 a[150:153], a[80:83], a[0:3], a[150:153]
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[0:3], v[0:3]
	ds_read_b128 a[0:3], v158 offset:33792
	ds_read_b128 a[4:7], v158 offset:33856
	ds_read_b128 a[8:11], v158 offset:33920
	ds_read_b128 a[12:15], v158 offset:33984
	ds_read_b128 a[16:19], v158 offset:34304
	ds_read_b128 a[20:23], v158 offset:34368
	ds_read_b128 a[24:27], v158 offset:34432
	ds_read_b128 a[28:31], v158 offset:34496
	ds_read_b128 a[32:35], v158 offset:50688
	ds_read_b128 a[36:39], v158 offset:50752
	ds_read_b128 a[40:43], v158 offset:50816
	ds_read_b128 a[44:47], v158 offset:50880
	ds_read_b128 a[48:51], v158 offset:51200
	ds_read_b128 a[52:55], v158 offset:51264
	ds_read_b128 a[56:59], v158 offset:51328
	ds_read_b128 a[60:63], v158 offset:51392
	ds_read_b128 a[64:67], v151
	ds_read_b128 a[68:71], v151 offset:64
	ds_read_b128 a[72:75], v151 offset:128
	ds_read_b128 a[76:79], v151 offset:192
	ds_read_b128 a[80:83], v151 offset:512
	ds_read_b128 a[84:87], v151 offset:576
	ds_read_b128 a[88:91], v151 offset:640
	ds_read_b128 a[92:95], v151 offset:704
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	v_accvgpr_read_b32 v4, a222
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 a[96:99], a[72:75], a[0:3], a[96:99]
	v_cndmask_b32_e32 v4, v4, v9, vcc
	s_mov_b32 m0, s35
	v_accvgpr_write_b32 a101, v4
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 a[96:99], a[76:79], a[4:7], a[96:99]
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	v_mov_b32_e32 v4, v159
	v_cndmask_b32_e32 v4, v4, v9, vcc
	v_mfma_f32_16x16x32_f16 v[246:249], a[64:67], a[8:11], v[18:21]
	s_mov_b32 m0, s36
	v_accvgpr_write_b32 a124, v4
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	v_accvgpr_read_b32 v18, a116
	v_accvgpr_read_b32 v19, a117
	v_accvgpr_read_b32 v20, a118
	v_accvgpr_read_b32 v21, a119
	v_accvgpr_read_b32 v149, a99
	v_accvgpr_read_b32 v148, a98
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[226:229], a[88:91], a[8:11], v[18:21]
	v_accvgpr_read_b32 v147, a97
	v_accvgpr_read_b32 v146, a96
	v_accvgpr_mov_b32 a96, a182
	v_accvgpr_read_b32 v18, a112
	v_accvgpr_read_b32 v19, a113
	v_accvgpr_read_b32 v20, a114
	v_accvgpr_read_b32 v21, a115
	v_accvgpr_mov_b32 a97, a183
	v_accvgpr_mov_b32 a98, a184
	v_mfma_f32_16x16x32_f16 v[222:225], a[64:67], a[16:19], v[18:21]
	v_accvgpr_mov_b32 a99, a185
	v_mov_b32_e32 v4, v161
	v_cndmask_b32_e32 v4, v4, v9, vcc
	v_accvgpr_read_b32 v18, a108
	v_accvgpr_read_b32 v19, a109
	v_accvgpr_read_b32 v20, a110
	v_accvgpr_read_b32 v21, a111
	v_mfma_f32_16x16x32_f16 a[96:99], a[64:67], a[48:51], a[96:99]
	v_accvgpr_read_b32 v6, a154
	v_accvgpr_read_b32 v7, a155
	v_accvgpr_read_b32 v8, a156
	v_mfma_f32_16x16x32_f16 v[218:221], a[72:75], a[16:19], v[18:21]
	v_accvgpr_read_b32 v9, a157
	s_mov_b32 m0, s37
	v_accvgpr_mov_b32 a116, a138
	v_accvgpr_read_b32 v18, a120
	v_accvgpr_read_b32 v19, a121
	v_accvgpr_read_b32 v20, a122
	v_accvgpr_read_b32 v21, a123
	v_mfma_f32_16x16x32_f16 v[6:9], a[64:67], a[0:3], v[6:9]
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_mov_b32 m0, s38
	s_waitcnt vmcnt(16)
	v_mfma_f32_16x16x32_f16 v[214:217], a[80:83], a[16:19], v[18:21]
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v28, s[56:59], 0 offen lds
	v_accvgpr_read_b32 v18, a252
	v_accvgpr_read_b32 v19, a253
	v_accvgpr_read_b32 v20, a254
	v_accvgpr_read_b32 v21, a255
	s_mov_b32 m0, s39
	v_mfma_f32_16x16x32_f16 a[224:227], a[80:83], a[0:3], a[224:227]
	buffer_load_dwordx4 v36, s[56:59], 0 offen lds
	s_mov_b32 m0, s40
	v_accvgpr_mov_b32 a120, a134
	v_mfma_f32_16x16x32_f16 v[210:213], a[88:91], a[16:19], v[18:21]
	buffer_load_dwordx4 v37, s[56:59], 0 offen lds
	s_mov_b32 m0, s41
	v_accvgpr_mov_b32 a121, a135
	v_accvgpr_read_b32 v18, a248
	v_accvgpr_read_b32 v19, a249
	v_accvgpr_read_b32 v20, a250
	v_accvgpr_read_b32 v21, a251
	v_mfma_f32_16x16x32_f16 a[248:251], a[68:71], a[52:55], a[96:99]
	buffer_load_dwordx4 v22, s[56:59], 0 offen lds
	s_mov_b32 m0, s42
	v_accvgpr_mov_b32 a122, a136
	v_accvgpr_mov_b32 a96, a186
	v_accvgpr_mov_b32 a97, a187
	v_accvgpr_mov_b32 a98, a188
	v_accvgpr_mov_b32 a99, a189
	v_mfma_f32_16x16x32_f16 v[206:209], a[64:67], a[24:27], v[18:21]
	buffer_load_dwordx4 v254, s[56:59], 0 offen lds
	s_mov_b32 m0, s43
	v_mov_b32_e32 v254, v34
	v_mfma_f32_16x16x32_f16 a[96:99], a[72:75], a[48:51], a[96:99]
	v_accvgpr_read_b32 v18, a244
	v_accvgpr_read_b32 v19, a245
	v_accvgpr_read_b32 v20, a246
	v_accvgpr_read_b32 v21, a247
	v_mfma_f32_16x16x32_f16 a[244:247], a[76:79], a[52:55], a[96:99]
	buffer_load_dwordx4 v255, s[56:59], 0 offen lds
	s_mov_b32 m0, s44
	v_mov_b32_e32 v255, v35
	v_mfma_f32_16x16x32_f16 v[202:205], a[72:75], a[24:27], v[18:21]
	v_accvgpr_mov_b32 a96, a190
	v_accvgpr_mov_b32 a97, a191
	v_accvgpr_mov_b32 a98, a192
	v_accvgpr_read_b32 v18, a240
	v_accvgpr_read_b32 v19, a241
	v_accvgpr_read_b32 v20, a242
	v_accvgpr_read_b32 v21, a243
	v_accvgpr_mov_b32 a99, a193
	v_mfma_f32_16x16x32_f16 a[232:235], a[88:91], a[0:3], a[232:235]
	buffer_load_dwordx4 v16, s[56:59], 0 offen lds
	s_mov_b32 m0, s45
	v_accvgpr_mov_b32 a123, a137
	v_mfma_f32_16x16x32_f16 v[198:201], a[80:83], a[24:27], v[18:21]
	buffer_load_dwordx4 v11, s[56:59], 0 offen lds
	s_mov_b32 m0, s46
	v_accvgpr_mov_b32 a117, a139
	v_accvgpr_read_b32 v18, a236
	v_mfma_f32_16x16x32_f16 a[96:99], a[80:83], a[48:51], a[96:99]
	v_accvgpr_read_b32 v19, a237
	v_accvgpr_read_b32 v20, a238
	v_accvgpr_read_b32 v21, a239
	v_mfma_f32_16x16x32_f16 a[240:243], a[84:87], a[52:55], a[96:99]
	buffer_load_dwordx4 v29, s[16:19], 0 offen lds
	s_mov_b32 m0, s47
	v_accvgpr_mov_b32 a118, a140
	v_mfma_f32_16x16x32_f16 v[194:197], a[88:91], a[24:27], v[18:21]
	v_accvgpr_mov_b32 a96, a194
	v_accvgpr_mov_b32 a97, a195
	v_accvgpr_mov_b32 a98, a196
	v_accvgpr_read_b32 v18, a218
	v_accvgpr_read_b32 v19, a219
	v_accvgpr_read_b32 v20, a220
	v_accvgpr_read_b32 v21, a221
	v_accvgpr_mov_b32 a99, a197
	v_mfma_f32_16x16x32_f16 v[250:253], a[72:75], a[8:11], v[250:253]
	buffer_load_dwordx4 v17, s[16:19], 0 offen lds
	s_mov_b32 m0, s48
	v_accvgpr_mov_b32 a119, a141
	v_mfma_f32_16x16x32_f16 v[190:193], a[64:67], a[32:35], v[18:21]
	buffer_load_dwordx4 v10, s[16:19], 0 offen lds
	s_mov_b32 m0, s49
	v_accvgpr_write_b32 a125, v4
	v_accvgpr_read_b32 v18, a214
	v_accvgpr_read_b32 v19, a215
	v_accvgpr_read_b32 v20, a216
	v_accvgpr_read_b32 v21, a217
	v_mfma_f32_16x16x32_f16 a[96:99], a[88:91], a[48:51], a[96:99]
	buffer_load_dwordx4 v23, s[16:19], 0 offen lds
	s_and_b32 s17, s21, 0xffff
	s_mov_b32 s16, s20
	v_mfma_f32_16x16x32_f16 v[186:189], a[72:75], a[32:35], v[18:21]
	s_mov_b32 m0, s22
	v_accvgpr_read_b32 v4, a107
	s_add_u32 s2, s2, 0x100
	v_accvgpr_read_b32 v18, a158
	v_accvgpr_read_b32 v19, a159
	v_accvgpr_read_b32 v20, a160
	v_accvgpr_read_b32 v21, a161
	v_mfma_f32_16x16x32_f16 a[236:239], a[92:95], a[52:55], a[96:99]
	v_accvgpr_mov_b32 a156, a178
	v_accvgpr_mov_b32 a157, a179
	v_accvgpr_mov_b32 a158, a180
	v_mfma_f32_16x16x32_f16 v[182:185], a[80:83], a[32:35], v[18:21]
	v_accvgpr_mov_b32 a96, a198
	v_accvgpr_mov_b32 a97, a199
	v_accvgpr_mov_b32 a98, a200
	v_accvgpr_read_b32 v18, a162
	v_accvgpr_read_b32 v19, a163
	v_accvgpr_read_b32 v20, a164
	v_accvgpr_read_b32 v21, a165
	v_accvgpr_mov_b32 a99, a201
	v_accvgpr_mov_b32 a160, a174
	v_mfma_f32_16x16x32_f16 v[178:181], a[88:91], a[32:35], v[18:21]
	v_accvgpr_mov_b32 a161, a175
	v_accvgpr_mov_b32 a162, a176
	v_accvgpr_mov_b32 a163, a177
	v_accvgpr_read_b32 v18, a166
	v_accvgpr_read_b32 v19, a167
	v_accvgpr_read_b32 v20, a168
	v_accvgpr_read_b32 v21, a169
	v_accvgpr_mov_b32 a164, a170
	v_accvgpr_mov_b32 a165, a171
	v_mfma_f32_16x16x32_f16 v[174:177], a[64:67], a[40:43], v[18:21]
	v_accvgpr_mov_b32 a166, a172
	v_accvgpr_mov_b32 a167, a173
	v_accvgpr_mov_b32 a159, a181
	v_mfma_f32_16x16x32_f16 a[64:67], a[64:67], a[56:59], a[96:99]
	v_accvgpr_read_b32 v18, a210
	v_accvgpr_read_b32 v19, a211
	v_accvgpr_read_b32 v20, a212
	v_mfma_f32_16x16x32_f16 a[112:115], a[68:71], a[60:63], a[64:67]
	v_accvgpr_read_b32 v21, a213
	s_addc_u32 s3, s3, 0
	v_accvgpr_read_b32 v29, a100
	v_mfma_f32_16x16x32_f16 a[228:231], a[80:83], a[8:11], a[228:231]
	v_accvgpr_mov_b32 a64, a202
	v_accvgpr_mov_b32 a65, a203
	v_accvgpr_mov_b32 a66, a204
	v_accvgpr_mov_b32 a67, a205
	v_mfma_f32_16x16x32_f16 a[164:167], a[72:75], a[40:43], a[164:167]
	s_cmp_lt_i32 s52, s9
	v_mfma_f32_16x16x32_f16 a[64:67], a[72:75], a[56:59], a[64:67]
	v_mfma_f32_16x16x32_f16 a[108:111], a[76:79], a[60:63], a[64:67]
	v_mfma_f32_16x16x32_f16 a[160:163], a[80:83], a[40:43], a[160:163]
	s_nop 5
	v_accvgpr_mov_b32 a64, a206
	v_accvgpr_mov_b32 a65, a207
	v_accvgpr_mov_b32 a66, a208
	v_accvgpr_mov_b32 a67, a209
	v_mfma_f32_16x16x32_f16 a[156:159], a[88:91], a[40:43], a[156:159]
	s_nop 0
	v_mfma_f32_16x16x32_f16 a[64:67], a[80:83], a[56:59], a[64:67]
	v_mfma_f32_16x16x32_f16 v[130:133], a[88:91], a[56:59], v[18:21]
	v_accvgpr_read_b32 v16, a130
	v_accvgpr_read_b32 v17, a131
	v_mfma_f32_16x16x32_f16 v[6:9], a[68:71], a[4:7], v[6:9]
	v_accvgpr_read_b32 v18, a132
	v_accvgpr_read_b32 v19, a133
	v_mfma_f32_16x16x32_f16 a[224:227], a[84:87], a[4:7], a[224:227]
	v_mfma_f32_16x16x32_f16 a[232:235], a[92:95], a[4:7], a[232:235]
	v_mfma_f32_16x16x32_f16 v[244:247], a[68:71], a[12:15], v[246:249]
	v_mfma_f32_16x16x32_f16 v[248:251], a[76:79], a[12:15], v[250:253]
	v_mfma_f32_16x16x32_f16 a[228:231], a[84:87], a[12:15], a[228:231]
	v_mfma_f32_16x16x32_f16 v[236:239], a[92:95], a[12:15], v[226:229]
	v_mfma_f32_16x16x32_f16 v[232:235], a[68:71], a[20:23], v[222:225]
	v_mfma_f32_16x16x32_f16 v[228:231], a[76:79], a[20:23], v[218:221]
	v_mfma_f32_16x16x32_f16 v[224:227], a[84:87], a[20:23], v[214:217]
	v_mfma_f32_16x16x32_f16 v[220:223], a[92:95], a[20:23], v[210:213]
	v_mfma_f32_16x16x32_f16 v[216:219], a[68:71], a[28:31], v[206:209]
	v_mfma_f32_16x16x32_f16 v[212:215], a[76:79], a[28:31], v[202:205]
	v_mfma_f32_16x16x32_f16 v[208:211], a[84:87], a[28:31], v[198:201]
	v_mfma_f32_16x16x32_f16 v[204:207], a[92:95], a[28:31], v[194:197]
	v_mfma_f32_16x16x32_f16 v[200:203], a[68:71], a[36:39], v[190:193]
	v_mfma_f32_16x16x32_f16 v[196:199], a[76:79], a[36:39], v[186:189]
	v_mfma_f32_16x16x32_f16 v[192:195], a[84:87], a[36:39], v[182:185]
	v_mfma_f32_16x16x32_f16 v[188:191], a[92:95], a[36:39], v[178:181]
	v_mfma_f32_16x16x32_f16 v[184:187], a[68:71], a[44:47], v[174:177]
	v_mfma_f32_16x16x32_f16 a[164:167], a[76:79], a[44:47], a[164:167]
	v_mfma_f32_16x16x32_f16 a[160:163], a[84:87], a[44:47], a[160:163]
	v_mfma_f32_16x16x32_f16 a[252:255], a[92:95], a[44:47], a[156:159]
	v_mfma_f32_16x16x32_f16 a[102:105], a[84:87], a[60:63], a[64:67]
	v_mfma_f32_16x16x32_f16 v[130:133], a[92:95], a[60:63], v[130:133]
	s_nop 1
	ds_read_b128 a[64:67], v152
	ds_read_b128 a[68:71], v152 offset:64
	ds_read_b128 a[72:75], v152 offset:128
	ds_read_b128 a[76:79], v152 offset:192
	ds_read_b128 a[80:83], v152 offset:512
	ds_read_b128 a[84:87], v152 offset:576
	ds_read_b128 a[88:91], v152 offset:640
	ds_read_b128 a[92:95], v152 offset:704
	s_waitcnt vmcnt(16) lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[126:129], a[64:67], a[0:3], v[126:129]
	s_barrier
	v_mfma_f32_16x16x32_f16 v[122:125], a[72:75], a[0:3], v[122:125]
	v_mfma_f32_16x16x32_f16 v[118:121], a[80:83], a[0:3], v[118:121]
	v_mfma_f32_16x16x32_f16 v[114:117], a[88:91], a[0:3], v[114:117]
	v_accvgpr_mov_b32 a0, a126
	v_accvgpr_mov_b32 a1, a127
	v_accvgpr_mov_b32 a2, a128
	v_mfma_f32_16x16x32_f16 v[34:37], a[88:91], a[40:43], v[16:19]
	v_accvgpr_mov_b32 a3, a129
	s_nop 1
	v_accvgpr_read_b32 v16, a142
	v_accvgpr_read_b32 v17, a143
	v_accvgpr_read_b32 v18, a144
	v_accvgpr_read_b32 v19, a145
	v_mfma_f32_16x16x32_f16 a[0:3], a[64:67], a[40:43], a[0:3]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[16:19], a[88:91], a[48:51], v[16:19]
	v_mfma_f32_16x16x32_f16 a[194:197], a[68:71], a[44:47], a[0:3]
	v_mfma_f32_16x16x32_f16 v[24:27], a[92:95], a[52:55], v[16:19]
	s_nop 3
	v_accvgpr_mov_b32 a0, a150
	v_accvgpr_mov_b32 a1, a151
	v_accvgpr_mov_b32 a2, a152
	v_accvgpr_read_b32 v16, a146
	v_accvgpr_read_b32 v17, a147
	v_accvgpr_read_b32 v18, a148
	v_accvgpr_read_b32 v19, a149
	v_accvgpr_mov_b32 a3, a153
	v_mfma_f32_16x16x32_f16 v[110:113], a[64:67], a[8:11], v[110:113]
	v_mfma_f32_16x16x32_f16 v[106:109], a[72:75], a[8:11], v[106:109]
	v_mfma_f32_16x16x32_f16 v[102:105], a[80:83], a[8:11], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[88:91], a[8:11], v[98:101]
	v_mfma_f32_16x16x32_f16 v[94:97], a[64:67], a[16:19], v[94:97]
	v_mfma_f32_16x16x32_f16 v[90:93], a[72:75], a[16:19], v[90:93]
	v_mfma_f32_16x16x32_f16 v[86:89], a[80:83], a[16:19], v[86:89]
	v_mfma_f32_16x16x32_f16 v[82:85], a[88:91], a[16:19], v[82:85]
	v_mfma_f32_16x16x32_f16 v[78:81], a[64:67], a[24:27], v[78:81]
	v_mfma_f32_16x16x32_f16 v[74:77], a[72:75], a[24:27], v[74:77]
	v_mfma_f32_16x16x32_f16 v[70:73], a[80:83], a[24:27], v[70:73]
	v_mfma_f32_16x16x32_f16 v[66:69], a[88:91], a[24:27], v[66:69]
	v_mfma_f32_16x16x32_f16 v[62:65], a[64:67], a[32:35], v[62:65]
	v_mfma_f32_16x16x32_f16 v[58:61], a[72:75], a[32:35], v[58:61]
	v_mfma_f32_16x16x32_f16 v[54:57], a[80:83], a[32:35], v[54:57]
	v_mfma_f32_16x16x32_f16 v[50:53], a[88:91], a[32:35], v[50:53]
	v_mfma_f32_16x16x32_f16 v[42:45], a[72:75], a[40:43], v[42:45]
	v_mfma_f32_16x16x32_f16 v[38:41], a[80:83], a[40:43], v[38:41]
	v_mfma_f32_16x16x32_f16 v[30:33], a[64:67], a[48:51], v[30:33]
	v_mfma_f32_16x16x32_f16 a[120:123], a[72:75], a[48:51], a[120:123]
	v_mfma_f32_16x16x32_f16 a[116:119], a[80:83], a[48:51], a[116:119]
	v_mfma_f32_16x16x32_f16 v[16:19], a[64:67], a[56:59], v[16:19]
	v_mfma_f32_16x16x32_f16 v[10:13], a[72:75], a[56:59], v[12:15]
	v_mfma_f32_16x16x32_f16 a[0:3], a[80:83], a[56:59], a[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[56:59], v[0:3]
	v_mfma_f32_16x16x32_f16 v[126:129], a[68:71], a[4:7], v[126:129]
	v_mfma_f32_16x16x32_f16 v[122:125], a[76:79], a[4:7], v[122:125]
	v_mfma_f32_16x16x32_f16 v[118:121], a[84:87], a[4:7], v[118:121]
	v_mfma_f32_16x16x32_f16 v[114:117], a[92:95], a[4:7], v[114:117]
	v_mfma_f32_16x16x32_f16 v[110:113], a[68:71], a[12:15], v[110:113]
	v_mfma_f32_16x16x32_f16 v[106:109], a[76:79], a[12:15], v[106:109]
	v_mfma_f32_16x16x32_f16 v[102:105], a[84:87], a[12:15], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[92:95], a[12:15], v[98:101]
	v_mfma_f32_16x16x32_f16 v[94:97], a[68:71], a[20:23], v[94:97]
	v_mfma_f32_16x16x32_f16 v[90:93], a[76:79], a[20:23], v[90:93]
	v_mfma_f32_16x16x32_f16 v[86:89], a[84:87], a[20:23], v[86:89]
	v_mfma_f32_16x16x32_f16 v[82:85], a[92:95], a[20:23], v[82:85]
	v_mfma_f32_16x16x32_f16 v[78:81], a[68:71], a[28:31], v[78:81]
	v_mfma_f32_16x16x32_f16 v[74:77], a[76:79], a[28:31], v[74:77]
	v_mfma_f32_16x16x32_f16 v[70:73], a[84:87], a[28:31], v[70:73]
	v_mfma_f32_16x16x32_f16 v[66:69], a[92:95], a[28:31], v[66:69]
	v_mfma_f32_16x16x32_f16 v[62:65], a[68:71], a[36:39], v[62:65]
	v_mfma_f32_16x16x32_f16 v[58:61], a[76:79], a[36:39], v[58:61]
	v_mfma_f32_16x16x32_f16 v[54:57], a[84:87], a[36:39], v[54:57]
	v_mfma_f32_16x16x32_f16 v[50:53], a[92:95], a[36:39], v[50:53]
	v_mfma_f32_16x16x32_f16 v[42:45], a[76:79], a[44:47], v[42:45]
	v_mfma_f32_16x16x32_f16 v[38:41], a[84:87], a[44:47], v[38:41]
	v_mfma_f32_16x16x32_f16 v[46:49], a[92:95], a[44:47], v[34:37]
	v_mfma_f32_16x16x32_f16 v[136:139], a[68:71], a[52:55], v[30:33]
	v_mfma_f32_16x16x32_f16 a[120:123], a[76:79], a[52:55], a[120:123]
	v_mfma_f32_16x16x32_f16 a[116:119], a[84:87], a[52:55], a[116:119]
	v_mfma_f32_16x16x32_f16 v[18:21], a[68:71], a[60:63], v[16:19]
	v_mfma_f32_16x16x32_f16 v[14:17], a[76:79], a[60:63], v[10:13]
	v_mfma_f32_16x16x32_f16 a[96:99], a[84:87], a[60:63], a[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[60:63], v[0:3]
	ds_read_b128 a[60:63], v5
	ds_read_b128 a[56:59], v5 offset:64
	ds_read_b128 a[48:51], v5 offset:128
	ds_read_b128 a[52:55], v5 offset:192
	ds_read_b128 a[44:47], v5 offset:512
	ds_read_b128 a[40:43], v5 offset:576
	ds_read_b128 a[36:39], v5 offset:640
	ds_read_b128 a[32:35], v5 offset:704
	ds_read_b128 a[28:31], v5 offset:16896
	ds_read_b128 a[24:27], v5 offset:16960
	ds_read_b128 a[20:23], v5 offset:17024
	ds_read_b128 a[16:19], v5 offset:17088
	ds_read_b128 a[12:15], v5 offset:17408
	ds_read_b128 a[8:11], v5 offset:17472
	ds_read_b128 a[4:7], v5 offset:17536
	ds_read_b128 a[0:3], v5 offset:17600
	ds_read_b128 a[92:95], v153
	ds_read_b128 a[88:91], v153 offset:64
	ds_read_b128 a[84:87], v153 offset:128
	ds_read_b128 a[80:83], v153 offset:192
	ds_read_b128 a[76:79], v153 offset:512
	ds_read_b128 a[72:75], v153 offset:576
	ds_read_b128 a[68:71], v153 offset:640
	ds_read_b128 a[64:67], v153 offset:704
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_mov_b32 m0, s23
	v_accvgpr_read_b32 v4, a101
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_mov_b32 m0, s50
	v_accvgpr_read_b32 v4, a124
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_mov_b32 m0, s51
	v_accvgpr_read_b32 v4, a125
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_cbranch_scc1 .LBB0_2
; %bb.3:                                ; %Flow
	v_accvgpr_read_b32 v28, a194
	v_mov_b64_e32 v[242:243], v[148:149]
	v_mov_b32_e32 v22, v143
	v_accvgpr_read_b32 v29, a195
	v_accvgpr_read_b32 v30, a196
	v_accvgpr_read_b32 v31, a197
	v_mov_b32_e32 v23, v142
	v_mov_b32_e32 v36, v141
	v_mov_b32_e32 v37, v140
	v_mov_b64_e32 v[240:241], v[146:147]
	s_branch .LBB0_5
.LBB0_4:
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v4, v3
	v_accvgpr_write_b32 a231, v7
	v_accvgpr_write_b32 a235, v7
	v_accvgpr_write_b32 a227, v7
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v0, v3
	v_accvgpr_write_b32 a99, v3
	v_accvgpr_write_b32 a98, v3
	v_accvgpr_write_b32 a97, v3
	v_accvgpr_write_b32 a96, v3
	v_mov_b32_e32 v17, v3
	v_mov_b32_e32 v16, v3
	v_mov_b32_e32 v15, v3
	v_mov_b32_e32 v14, v3
	v_mov_b32_e32 v21, v3
	v_mov_b32_e32 v20, v3
	v_mov_b32_e32 v19, v3
	v_mov_b32_e32 v18, v3
	v_mov_b32_e32 v27, v3
	v_mov_b32_e32 v26, v3
	v_mov_b32_e32 v25, v3
	v_mov_b32_e32 v24, v3
	v_accvgpr_write_b32 a119, v3
	v_accvgpr_write_b32 a118, v3
	v_accvgpr_write_b32 a117, v3
	v_accvgpr_write_b32 a116, v3
	v_accvgpr_write_b32 a123, v3
	v_accvgpr_write_b32 a122, v3
	v_accvgpr_write_b32 a121, v3
	v_accvgpr_write_b32 a120, v3
	v_mov_b32_e32 v139, v3
	v_mov_b32_e32 v138, v3
	v_mov_b32_e32 v137, v3
	v_mov_b32_e32 v136, v3
	v_mov_b32_e32 v49, v3
	v_mov_b32_e32 v48, v3
	v_mov_b32_e32 v47, v3
	v_mov_b32_e32 v46, v3
	v_mov_b32_e32 v41, v3
	v_mov_b32_e32 v40, v3
	v_mov_b32_e32 v39, v3
	v_mov_b32_e32 v38, v3
	v_mov_b32_e32 v45, v3
	v_mov_b32_e32 v44, v3
	v_mov_b32_e32 v43, v3
	v_mov_b32_e32 v42, v3
	v_mov_b32_e32 v31, v3
	v_mov_b32_e32 v30, v3
	v_mov_b32_e32 v29, v3
	v_mov_b32_e32 v28, v3
	v_mov_b32_e32 v53, v3
	v_mov_b32_e32 v52, v3
	v_mov_b32_e32 v51, v3
	v_mov_b32_e32 v50, v3
	v_mov_b32_e32 v57, v3
	v_mov_b32_e32 v56, v3
	v_mov_b32_e32 v55, v3
	v_mov_b32_e32 v54, v3
	v_mov_b32_e32 v61, v3
	v_mov_b32_e32 v60, v3
	v_mov_b32_e32 v59, v3
	v_mov_b32_e32 v58, v3
	v_mov_b32_e32 v65, v3
	v_mov_b32_e32 v64, v3
	v_mov_b32_e32 v63, v3
	v_mov_b32_e32 v62, v3
	v_mov_b32_e32 v69, v3
	v_mov_b32_e32 v68, v3
	v_mov_b32_e32 v67, v3
	v_mov_b32_e32 v66, v3
	v_mov_b32_e32 v73, v3
	v_mov_b32_e32 v72, v3
	v_mov_b32_e32 v71, v3
	v_mov_b32_e32 v70, v3
	v_mov_b32_e32 v77, v3
	v_mov_b32_e32 v76, v3
	v_mov_b32_e32 v75, v3
	v_mov_b32_e32 v74, v3
	v_mov_b32_e32 v81, v3
	v_mov_b32_e32 v80, v3
	v_mov_b32_e32 v79, v3
	v_mov_b32_e32 v78, v3
	v_mov_b32_e32 v85, v3
	v_mov_b32_e32 v84, v3
	v_mov_b32_e32 v83, v3
	v_mov_b32_e32 v82, v3
	v_mov_b32_e32 v89, v3
	v_mov_b32_e32 v88, v3
	v_mov_b32_e32 v87, v3
	v_mov_b32_e32 v86, v3
	v_mov_b32_e32 v93, v3
	v_mov_b32_e32 v92, v3
	v_mov_b32_e32 v91, v3
	v_mov_b32_e32 v90, v3
	v_mov_b32_e32 v97, v3
	v_mov_b32_e32 v96, v3
	v_mov_b32_e32 v95, v3
	v_mov_b32_e32 v94, v3
	v_mov_b32_e32 v101, v3
	v_mov_b32_e32 v100, v3
	v_mov_b32_e32 v99, v3
	v_mov_b32_e32 v98, v3
	v_mov_b32_e32 v105, v3
	v_mov_b32_e32 v104, v3
	v_mov_b32_e32 v103, v3
	v_mov_b32_e32 v102, v3
	v_mov_b32_e32 v109, v3
	v_mov_b32_e32 v108, v3
	v_mov_b32_e32 v107, v3
	v_mov_b32_e32 v106, v3
	v_mov_b32_e32 v113, v3
	v_mov_b32_e32 v112, v3
	v_mov_b32_e32 v111, v3
	v_mov_b32_e32 v110, v3
	v_mov_b32_e32 v117, v3
	v_mov_b32_e32 v116, v3
	v_mov_b32_e32 v115, v3
	v_mov_b32_e32 v114, v3
	v_mov_b32_e32 v121, v3
	v_mov_b32_e32 v120, v3
	v_mov_b32_e32 v119, v3
	v_mov_b32_e32 v118, v3
	v_mov_b32_e32 v125, v3
	v_mov_b32_e32 v124, v3
	v_mov_b32_e32 v123, v3
	v_mov_b32_e32 v122, v3
	v_mov_b32_e32 v129, v3
	v_mov_b32_e32 v128, v3
	v_mov_b32_e32 v127, v3
	v_mov_b32_e32 v126, v3
	v_mov_b32_e32 v133, v3
	v_mov_b32_e32 v132, v3
	v_mov_b32_e32 v131, v3
	v_mov_b32_e32 v130, v3
	v_accvgpr_write_b32 a105, v3
	v_accvgpr_write_b32 a104, v3
	v_accvgpr_write_b32 a103, v3
	v_accvgpr_write_b32 a102, v3
	v_accvgpr_write_b32 a111, v3
	v_accvgpr_write_b32 a110, v3
	v_accvgpr_write_b32 a109, v3
	v_accvgpr_write_b32 a108, v3
	v_accvgpr_write_b32 a115, v3
	v_accvgpr_write_b32 a114, v3
	v_accvgpr_write_b32 a113, v3
	v_accvgpr_write_b32 a112, v3
	v_accvgpr_write_b32 a239, v3
	v_accvgpr_write_b32 a238, v3
	v_accvgpr_write_b32 a237, v3
	v_accvgpr_write_b32 a236, v3
	v_accvgpr_write_b32 a243, v3
	v_accvgpr_write_b32 a242, v3
	v_accvgpr_write_b32 a241, v3
	v_accvgpr_write_b32 a240, v3
	v_accvgpr_write_b32 a247, v3
	v_accvgpr_write_b32 a246, v3
	v_accvgpr_write_b32 a245, v3
	v_accvgpr_write_b32 a244, v3
	v_accvgpr_write_b32 a251, v3
	v_accvgpr_write_b32 a250, v3
	v_accvgpr_write_b32 a249, v3
	v_accvgpr_write_b32 a248, v3
	v_accvgpr_write_b32 a255, v3
	v_accvgpr_write_b32 a254, v3
	v_accvgpr_write_b32 a253, v3
	v_accvgpr_write_b32 a252, v3
	v_accvgpr_write_b32 a163, v3
	v_accvgpr_write_b32 a162, v3
	v_accvgpr_write_b32 a161, v3
	v_accvgpr_write_b32 a160, v3
	v_accvgpr_write_b32 a167, v3
	v_accvgpr_write_b32 a166, v3
	v_accvgpr_write_b32 a165, v3
	v_accvgpr_write_b32 a164, v3
	v_mov_b32_e32 v187, v3
	v_mov_b32_e32 v186, v3
	v_mov_b32_e32 v185, v3
	v_mov_b32_e32 v184, v3
	v_mov_b32_e32 v191, v3
	v_mov_b32_e32 v190, v3
	v_mov_b32_e32 v189, v3
	v_mov_b32_e32 v188, v3
	v_mov_b32_e32 v195, v3
	v_mov_b32_e32 v194, v3
	v_mov_b32_e32 v193, v3
	v_mov_b32_e32 v192, v3
	v_mov_b32_e32 v199, v3
	v_mov_b32_e32 v198, v3
	v_mov_b32_e32 v197, v3
	v_mov_b32_e32 v196, v3
	v_mov_b32_e32 v203, v3
	v_mov_b32_e32 v202, v3
	v_mov_b32_e32 v201, v3
	v_mov_b32_e32 v200, v3
	v_mov_b32_e32 v207, v3
	v_mov_b32_e32 v206, v3
	v_mov_b32_e32 v205, v3
	v_mov_b32_e32 v204, v3
	v_mov_b32_e32 v211, v3
	v_mov_b32_e32 v210, v3
	v_mov_b32_e32 v209, v3
	v_mov_b32_e32 v208, v3
	v_mov_b32_e32 v215, v3
	v_mov_b32_e32 v214, v3
	v_mov_b32_e32 v213, v3
	v_mov_b32_e32 v212, v3
	v_mov_b32_e32 v219, v3
	v_mov_b32_e32 v218, v3
	v_mov_b32_e32 v217, v3
	v_mov_b32_e32 v216, v3
	v_mov_b32_e32 v223, v3
	v_mov_b32_e32 v222, v3
	v_mov_b32_e32 v221, v3
	v_mov_b32_e32 v220, v3
	v_mov_b32_e32 v227, v3
	v_mov_b32_e32 v226, v3
	v_mov_b32_e32 v225, v3
	v_mov_b32_e32 v224, v3
	v_mov_b32_e32 v231, v3
	v_mov_b32_e32 v230, v3
	v_mov_b32_e32 v229, v3
	v_mov_b32_e32 v228, v3
	v_mov_b32_e32 v235, v3
	v_mov_b32_e32 v234, v3
	v_mov_b32_e32 v233, v3
	v_mov_b32_e32 v232, v3
	v_mov_b32_e32 v239, v3
	v_mov_b32_e32 v238, v3
	v_mov_b32_e32 v237, v3
	v_mov_b32_e32 v236, v3
	v_accvgpr_write_b32 a230, v6
	v_accvgpr_write_b32 a229, v5
	v_accvgpr_write_b32 a228, v4
	v_mov_b32_e32 v251, v3
	v_mov_b32_e32 v250, v3
	v_mov_b32_e32 v249, v3
	v_mov_b32_e32 v248, v3
	v_mov_b32_e32 v247, v3
	v_mov_b32_e32 v246, v3
	v_mov_b32_e32 v245, v3
	v_mov_b32_e32 v244, v3
	v_accvgpr_write_b32 a234, v6
	v_accvgpr_write_b32 a233, v5
	v_accvgpr_write_b32 a232, v4
	v_accvgpr_write_b32 a226, v6
	v_accvgpr_write_b32 a225, v5
	v_accvgpr_write_b32 a224, v4
	v_mov_b32_e32 v243, v3
	v_mov_b32_e32 v242, v3
	v_mov_b32_e32 v241, v3
	v_mov_b32_e32 v240, v3
	v_mov_b32_e32 v9, v3
	v_mov_b32_e32 v8, v3
.LBB0_5:                                ; %._crit_edge
	v_accvgpr_mov_b32 a131, a105
	v_accvgpr_mov_b32 a130, a104
	v_accvgpr_mov_b32 a129, a103
	v_accvgpr_mov_b32 a128, a102
	v_accvgpr_mov_b32 a103, a99
	v_accvgpr_mov_b32 a102, a98
	v_accvgpr_mov_b32 a101, a97
	v_accvgpr_mov_b32 a100, a96
	v_accvgpr_write_b32 a99, v3
	v_accvgpr_write_b32 a98, v2
	v_accvgpr_write_b32 a97, v1
	v_accvgpr_write_b32 a96, v0
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[60:63], v[6:9]
	s_mul_i32 s2, s1, s13
	s_ashr_i32 s3, s2, 31
	v_accvgpr_write_b32 a107, v17
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[48:51], v[248:251]
	s_lshl_b32 s19, s24, 6
	s_lshl_b64 s[2:3], s[2:3], 1
	v_accvgpr_write_b32 a106, v16
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[8:11], a[80:83], a[52:55], v[4:7]
	v_accvgpr_write_b32 a105, v15
	v_accvgpr_write_b32 a104, v14
	s_add_u32 s2, s6, s2
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[48:51], v[244:247]
	s_addc_u32 s3, s7, s3
	s_ashr_i32 s1, s0, 31
	v_accvgpr_mov_b32 a139, a115
	v_mfma_f32_16x16x32_f16 v[12:15], a[88:91], a[52:55], v[4:7]
	s_lshl_b64 s[0:1], s[0:1], 1
	v_accvgpr_mov_b32 a138, a114
	v_accvgpr_mov_b32 a137, a113
	v_accvgpr_mov_b32 a136, a112
	v_accvgpr_read_b32 v4, a232
	v_accvgpr_read_b32 v5, a233
	v_accvgpr_read_b32 v6, a234
	v_accvgpr_read_b32 v7, a235
	v_accvgpr_write_b32 a115, v27
	s_add_u32 s4, s2, s0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[60:63], v[4:7]
	v_accvgpr_mov_b32 a135, a111
	v_accvgpr_write_b32 a114, v26
	v_accvgpr_write_b32 a113, v25
	v_accvgpr_write_b32 a112, v24
	v_accvgpr_read_b32 v24, a228
	s_addc_u32 s6, s3, s1
	s_lshl_b32 s0, s13, 6
	v_accvgpr_mov_b32 a134, a110
	v_accvgpr_mov_b32 a133, a109
	v_accvgpr_mov_b32 a132, a108
	v_accvgpr_write_b32 a111, v21
	v_accvgpr_read_b32 v25, a229
	v_accvgpr_read_b32 v26, a230
	v_accvgpr_read_b32 v27, a231
	s_ashr_i32 s1, s0, 31
	v_accvgpr_write_b32 a110, v20
	v_accvgpr_write_b32 a109, v19
	v_accvgpr_write_b32 a108, v18
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[16:19], a[64:67], a[56:59], v[4:7]
	s_lshl_b64 s[0:1], s[0:1], 1
	v_accvgpr_write_b32 a216, v232
	v_accvgpr_write_b32 a212, v228
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[60:63], v[240:243]
	s_add_u32 s28, s4, s0
	v_accvgpr_write_b32 a217, v233
	v_accvgpr_write_b32 a218, v234
	v_mfma_f32_16x16x32_f16 v[242:245], a[76:79], a[48:51], v[24:27]
	v_accvgpr_write_b32 a219, v235
	v_accvgpr_write_b32 a213, v229
	v_accvgpr_write_b32 a214, v230
	v_accvgpr_read_b32 v24, a224
	v_accvgpr_read_b32 v25, a225
	v_accvgpr_read_b32 v26, a226
	v_accvgpr_read_b32 v27, a227
	v_accvgpr_write_b32 a215, v231
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[56:59], v[0:3]
	s_addc_u32 s18, s6, s1
	s_add_u32 s24, s28, s0
	s_addc_u32 s17, s18, s1
	v_mfma_f32_16x16x32_f16 v[230:233], a[76:79], a[60:63], v[24:27]
	v_accvgpr_write_b32 a220, v236
	s_add_u32 s20, s24, s0
	v_accvgpr_write_b32 a221, v237
	v_accvgpr_write_b32 a222, v238
	v_accvgpr_write_b32 a223, v239
	v_cvt_pk_f16_f32 v238, v0, v1
	v_cvt_pk_f16_f32 v239, v2, v3
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[56:59], v[230:233]
	s_addc_u32 s11, s17, s1
	s_add_u32 s16, s4, 0x100
	s_addc_u32 s10, s6, 0
	s_add_u32 s12, s28, 0x100
	s_addc_u32 s3, s18, 0
	v_mfma_f32_16x16x32_f16 v[4:7], a[80:83], a[56:59], v[4:7]
	s_add_u32 s8, s24, 0x100
	s_addc_u32 s2, s17, 0
	v_cvt_pk_f16_f32 v230, v0, v1
	v_cvt_pk_f16_f32 v231, v2, v3
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[52:55], v[242:245]
	s_add_u32 s0, s20, 0x100
	s_addc_u32 s1, s11, 0
	s_lshr_b32 s7, s9, 31
	v_and_b32_e32 v242, 63, v36
	v_cvt_pk_f16_f32 v235, v6, v7
	v_cvt_pk_f16_f32 v7, v18, v19
	s_add_i32 s7, s9, s7
	s_and_b32 s7, s7, -2
	v_cvt_pk_f16_f32 v232, v0, v1
	v_or_b32_e32 v0, s19, v242
	v_cvt_pk_f16_f32 v233, v2, v3
	v_lshrrev_b32_e32 v18, 4, v0
	v_accvgpr_read_b32 v0, a220
	v_accvgpr_read_b32 v1, a221
	v_accvgpr_read_b32 v2, a222
	v_accvgpr_read_b32 v3, a223
	v_accvgpr_write_b32 a208, v224
	s_sub_i32 s9, s9, s7
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[48:51], v[0:3]
	s_lshr_b32 s5, s5, 1
	v_accvgpr_write_b32 a209, v225
	v_accvgpr_write_b32 a210, v226
	v_accvgpr_write_b32 a211, v227
	v_and_b32_e32 v226, 1, v36
	s_and_b64 s[14:15], s[14:15], exec
	v_cvt_pk_f16_f32 v234, v4, v5
	v_lshlrev_b32_e32 v4, 8, v36
	v_lshlrev_b32_e32 v5, 12, v226
	v_and_b32_e32 v227, 16, v36
	s_movk_i32 s14, 0x2e00
	v_accvgpr_write_b32 a204, v220
	v_cvt_pk_f16_f32 v237, v10, v11
	v_lshlrev_b32_e32 v10, 4, v227
	s_cselect_b32 s7, 0, 0x80
	v_and_or_b32 v4, v4, s14, v5
	v_accvgpr_write_b32 a205, v221
	v_accvgpr_write_b32 a206, v222
	v_accvgpr_write_b32 a207, v223
	v_cvt_pk_f16_f32 v240, v12, v13
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[52:55], v[0:3]
	v_or3_b32 v222, s7, v10, v4
	v_accvgpr_read_b32 v10, a212
	v_accvgpr_read_b32 v11, a213
	v_accvgpr_read_b32 v12, a214
	v_accvgpr_read_b32 v13, a215
	v_cvt_pk_f16_f32 v236, v8, v9
	v_cvt_pk_f16_f32 v6, v16, v17
	v_mfma_f32_16x16x32_f16 v[10:13], a[84:87], a[44:47], v[10:13]
	v_cvt_pk_f16_f32 v8, v0, v1
	v_cvt_pk_f16_f32 v9, v2, v3
	v_accvgpr_read_b32 v0, a216
	v_accvgpr_read_b32 v1, a217
	v_accvgpr_read_b32 v2, a218
	v_accvgpr_read_b32 v3, a219
	v_cvt_pk_f16_f32 v241, v14, v15
	v_mfma_f32_16x16x32_f16 v[14:17], a[80:83], a[40:43], v[10:13]
	v_accvgpr_write_b32 a200, v216
	v_accvgpr_write_b32 a196, v212
	v_or_b32_e32 v19, 16, v18
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[44:47], v[0:3]
	v_accvgpr_read_b32 v10, a208
	v_accvgpr_read_b32 v11, a209
	v_accvgpr_read_b32 v12, a210
	v_accvgpr_read_b32 v13, a211
	v_mfma_f32_16x16x32_f16 v[2:5], a[88:91], a[40:43], v[0:3]
	v_or_b32_e32 v20, 48, v18
	v_accvgpr_write_b32 a201, v217
	v_accvgpr_write_b32 a202, v218
	v_mfma_f32_16x16x32_f16 v[10:13], a[76:79], a[44:47], v[10:13]
	v_mov_b32_e32 v0, 0x70
	v_bitop3_b32 v223, s5, v23, v0 bitop3:0x78
	s_movk_i32 s5, 0x50
	v_accvgpr_write_b32 a203, v219
	v_accvgpr_write_b32 a197, v213
	v_accvgpr_write_b32 a198, v214
	v_accvgpr_write_b32 a199, v215
	v_or_b32_e32 v224, v222, v223
	v_or_b32_e32 v1, 32, v18
	v_mul_lo_u32 v219, v18, s13
	v_mul_lo_u32 v214, v19, s13
	v_mul_lo_u32 v216, v20, s13
	v_mfma_f32_16x16x32_f16 v[18:21], a[72:75], a[40:43], v[10:13]
	v_add_u32_e32 v0, 0, v224
	v_mul_lo_u32 v215, v1, s13
	v_xad_u32 v1, v224, 16, 0
	v_bitop3_b32 v11, v222, s5, v223 bitop3:0x36
	v_xad_u32 v10, v224, 64, 0
	v_add_u32_e32 v11, 0, v11
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[238:241]
	ds_write_b128 v1, v[234:237]
	ds_write_b128 v10, v[230:233]
	ds_write_b128 v11, v[6:9]
	v_accvgpr_read_b32 v6, a200
	v_accvgpr_read_b32 v7, a201
	v_accvgpr_read_b32 v8, a202
	v_accvgpr_read_b32 v9, a203
	v_accvgpr_write_b32 a192, v208
	v_accvgpr_write_b32 a188, v204
	v_mfma_f32_16x16x32_f16 v[6:9], a[92:95], a[36:39], v[6:9]
	v_mov_b32_e32 v13, 0xe0
	v_accvgpr_write_b32 a193, v209
	v_accvgpr_write_b32 a194, v210
	v_accvgpr_write_b32 a195, v211
	v_accvgpr_write_b32 a189, v205
	v_accvgpr_write_b32 a190, v206
	v_accvgpr_write_b32 a191, v207
	v_mfma_f32_16x16x32_f16 v[206:209], a[88:91], a[32:35], v[6:9]
	v_accvgpr_read_b32 v24, a204
	v_accvgpr_read_b32 v25, a205
	v_accvgpr_read_b32 v26, a206
	v_bitop3_b32 v6, s19, v13, v242 bitop3:0xc8
	v_lshlrev_b32_e32 v13, 4, v6
	v_lshrrev_b32_e32 v217, 1, v6
	v_accvgpr_read_b32 v6, a196
	v_accvgpr_read_b32 v27, a207
	v_accvgpr_read_b32 v7, a197
	v_accvgpr_read_b32 v8, a198
	v_accvgpr_read_b32 v9, a199
	v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[44:47], v[24:27]
	v_accvgpr_write_b32 a184, v200
	v_accvgpr_write_b32 a180, v196
	v_accvgpr_write_b32 a185, v201
	v_mfma_f32_16x16x32_f16 v[6:9], a[84:87], a[36:39], v[6:9]
	v_accvgpr_read_b32 v24, a192
	v_accvgpr_read_b32 v25, a193
	v_accvgpr_read_b32 v26, a194
	v_accvgpr_read_b32 v27, a195
	v_accvgpr_write_b32 a186, v202
	v_accvgpr_write_b32 a187, v203
	v_accvgpr_write_b32 a181, v197
	v_accvgpr_write_b32 a182, v198
	v_accvgpr_write_b32 a183, v199
	v_and_b32_e32 v12, 0x70, v23
	v_mfma_f32_16x16x32_f16 v[198:201], a[76:79], a[36:39], v[24:27]
	v_accvgpr_write_b32 a176, v192
	v_lshlrev_b32_e32 v220, 8, v227
	v_bitop3_b32 v12, v13, v217, v12 bitop3:0x36
	v_accvgpr_read_b32 v24, a188
	v_lshl_add_u32 v13, v226, 13, 0
	v_accvgpr_read_b32 v25, a189
	v_accvgpr_read_b32 v26, a190
	v_accvgpr_read_b32 v27, a191
	v_accvgpr_write_b32 a177, v193
	v_accvgpr_write_b32 a178, v194
	v_accvgpr_write_b32 a179, v195
	v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[32:35], v[6:9]
	v_accvgpr_write_b32 a172, v188
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add3_u32 v6, v13, v220, v12
	v_mfma_f32_16x16x32_f16 v[194:197], a[68:71], a[36:39], v[24:27]
	ds_read_b128 v[220:223], v6
	ds_read_b128 v[226:229], v6 offset:128
	ds_read_b128 v[232:235], v6 offset:256
	ds_read_b128 v[236:239], v6 offset:384
	v_accvgpr_read_b32 v24, a184
	v_accvgpr_read_b32 v25, a185
	v_accvgpr_read_b32 v26, a186
	v_accvgpr_read_b32 v27, a187
	v_accvgpr_write_b32 a173, v189
	v_accvgpr_write_b32 a174, v190
	v_accvgpr_write_b32 a175, v191
	v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], a[32:35], v[198:201]
	v_accvgpr_write_b32 a168, v184
	v_accvgpr_write_b32 a169, v185
	v_accvgpr_write_b32 a170, v186
	v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[28:31], v[24:27]
	v_accvgpr_write_b32 a171, v187
	v_lshlrev_b32_e32 v218, 3, v37
	s_and_b32 s5, s6, 0xffff
	v_accvgpr_read_b32 v24, a180
	v_accvgpr_read_b32 v25, a181
	v_accvgpr_read_b32 v26, a182
	v_accvgpr_read_b32 v27, a183
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_mfma_f32_16x16x32_f16 v[186:189], a[84:87], a[28:31], v[24:27]
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v230, v220
	v_mov_b32_e32 v231, v221
	v_add_lshl_u32 v7, v219, v218, 1
	v_accvgpr_read_b32 v24, a176
	v_accvgpr_read_b32 v25, a177
	v_accvgpr_read_b32 v26, a178
	v_accvgpr_read_b32 v27, a179
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[230:233], v7, s[4:7], 0 offen
	v_mov_b32_e32 v224, v234
	v_mov_b32_e32 v225, v235
	v_add_lshl_u32 v8, v214, v218, 1
	v_mov_b32_e32 v234, v226
	v_mov_b32_e32 v235, v227
	v_add_lshl_u32 v9, v215, v218, 1
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v230, v238
	v_mov_b32_e32 v231, v239
	v_add_lshl_u32 v12, v216, v218, 1
	v_cvt_pk_f16_f32 v2, v2, v3
	v_cvt_pk_f16_f32 v3, v4, v5
	v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[28:31], v[24:27]
	v_cvt_pk_f16_f32 v18, v18, v19
	v_cvt_pk_f16_f32 v19, v20, v21
	v_cvt_pk_f16_f32 v4, v206, v207
	v_accvgpr_read_b32 v24, a172
	v_cvt_pk_f16_f32 v5, v208, v209
	v_cvt_pk_f16_f32 v20, v198, v199
	v_cvt_pk_f16_f32 v21, v200, v201
	buffer_store_dwordx4 v[222:225], v8, s[4:7], 0 offen
	buffer_store_dwordx4 v[234:237], v9, s[4:7], 0 offen
	buffer_store_dwordx4 v[228:231], v12, s[4:7], 0 offen
	v_cvt_pk_f16_f32 v14, v14, v15
	v_cvt_pk_f16_f32 v15, v16, v17
	v_accvgpr_read_b32 v25, a173
	v_accvgpr_read_b32 v26, a174
	v_accvgpr_read_b32 v27, a175
	v_cvt_pk_f16_f32 v16, v202, v203
	v_cvt_pk_f16_f32 v17, v204, v205
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[2:5]
	ds_write_b128 v1, v[14:17]
	ds_write_b128 v10, v[18:21]
	v_accvgpr_read_b32 v18, a252
	v_mfma_f32_16x16x32_f16 v[178:181], a[68:71], a[28:31], v[24:27]
	v_accvgpr_read_b32 v14, a160
	v_accvgpr_read_b32 v19, a253
	v_accvgpr_read_b32 v20, a254
	v_accvgpr_read_b32 v24, a168
	v_accvgpr_read_b32 v21, a255
	v_mfma_f32_16x16x32_f16 v[210:213], a[64:67], a[40:43], v[210:213]
	v_accvgpr_read_b32 v25, a169
	v_accvgpr_read_b32 v26, a170
	v_accvgpr_read_b32 v27, a171
	v_mfma_f32_16x16x32_f16 v[194:197], a[64:67], a[32:35], v[194:197]
	v_accvgpr_read_b32 v15, a161
	v_accvgpr_read_b32 v16, a162
	v_accvgpr_read_b32 v17, a163
	v_mfma_f32_16x16x32_f16 v[18:21], a[68:71], a[20:23], v[18:21]
	v_accvgpr_read_b32 v2, a164
	v_accvgpr_read_b32 v3, a165
	v_accvgpr_read_b32 v4, a166
	v_mfma_f32_16x16x32_f16 v[174:177], a[92:95], a[20:23], v[24:27]
	v_accvgpr_read_b32 v5, a167
	v_cvt_pk_f16_f32 v210, v210, v211
	v_cvt_pk_f16_f32 v211, v212, v213
	v_mfma_f32_16x16x32_f16 v[14:17], a[76:79], a[20:23], v[14:17]
	v_accvgpr_read_b32 v24, a248
	v_accvgpr_read_b32 v25, a249
	v_accvgpr_read_b32 v26, a250
	v_accvgpr_read_b32 v27, a251
	v_mfma_f32_16x16x32_f16 v[178:181], a[64:67], a[24:27], v[178:181]
	v_cvt_pk_f16_f32 v212, v194, v195
	v_cvt_pk_f16_f32 v213, v196, v197
	ds_write_b128 v11, v[210:213]
	v_mfma_f32_16x16x32_f16 v[158:161], a[92:95], a[12:15], v[24:27]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[194:197], v6
	ds_read_b128 v[200:203], v6 offset:128
	ds_read_b128 v[170:173], v6 offset:256
	ds_read_b128 v[206:209], v6 offset:384
	v_accvgpr_read_b32 v24, a244
	v_mfma_f32_16x16x32_f16 v[2:5], a[84:87], a[20:23], v[2:5]
	v_accvgpr_read_b32 v25, a245
	v_accvgpr_read_b32 v26, a246
	v_accvgpr_read_b32 v27, a247
	v_mfma_f32_16x16x32_f16 v[18:21], a[64:67], a[16:19], v[18:21]
	v_accvgpr_write_b32 a124, v136
	v_cvt_pk_f16_f32 v178, v178, v179
	v_cvt_pk_f16_f32 v179, v180, v181
	v_mfma_f32_16x16x32_f16 v[14:17], a[72:75], a[16:19], v[14:17]
	v_accvgpr_write_b32 a125, v137
	v_accvgpr_write_b32 a126, v138
	v_accvgpr_write_b32 a127, v139
	v_mfma_f32_16x16x32_f16 v[154:157], a[84:87], a[12:15], v[24:27]
	v_cvt_pk_f16_f32 v180, v18, v19
	v_cvt_pk_f16_f32 v181, v20, v21
	v_accvgpr_read_b32 v18, a136
	v_accvgpr_read_b32 v24, a240
	v_accvgpr_read_b32 v25, a241
	v_accvgpr_read_b32 v26, a242
	v_accvgpr_read_b32 v27, a243
	v_mfma_f32_16x16x32_f16 v[190:193], a[88:91], a[24:27], v[190:193]
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v198, v172
	v_mov_b32_e32 v199, v173
	v_cvt_pk_f16_f32 v172, v14, v15
	v_mfma_f32_16x16x32_f16 v[150:153], a[76:79], a[12:15], v[24:27]
	v_cvt_pk_f16_f32 v173, v16, v17
	v_accvgpr_read_b32 v14, a236
	v_accvgpr_read_b32 v19, a137
	v_accvgpr_read_b32 v24, a132
	v_mfma_f32_16x16x32_f16 v[186:189], a[80:83], a[24:27], v[186:189]
	v_accvgpr_read_b32 v25, a133
	v_accvgpr_read_b32 v26, a134
	v_accvgpr_read_b32 v27, a135
	v_mfma_f32_16x16x32_f16 v[182:185], a[72:75], a[24:27], v[182:185]
	v_accvgpr_read_b32 v20, a138
	v_accvgpr_read_b32 v21, a139
	v_accvgpr_read_b32 v15, a237
	v_mfma_f32_16x16x32_f16 v[174:177], a[88:91], a[16:19], v[174:177]
	v_accvgpr_read_b32 v16, a238
	v_accvgpr_read_b32 v17, a239
	s_and_b32 s29, s18, 0xffff
	v_mfma_f32_16x16x32_f16 v[2:5], a[80:83], a[16:19], v[2:5]
	s_mov_b32 s30, s6
	s_mov_b32 s31, s7
	v_mov_b32_e32 v168, v194
	v_mfma_f32_16x16x32_f16 v[138:141], a[84:87], a[4:7], v[24:27]
	v_mov_b32_e32 v169, v195
	v_mov_b32_e32 v204, v200
	v_mov_b32_e32 v205, v201
	v_accvgpr_read_b32 v24, a128
	v_mfma_f32_16x16x32_f16 v[18:21], a[92:95], a[4:7], v[18:21]
	v_accvgpr_read_b32 v25, a129
	v_accvgpr_read_b32 v26, a130
	v_accvgpr_read_b32 v27, a131
	v_mfma_f32_16x16x32_f16 v[14:17], a[68:71], a[12:15], v[14:17]
	buffer_store_dwordx4 v[168:171], v7, s[28:31], 0 offen
	buffer_store_dwordx4 v[196:199], v8, s[28:31], 0 offen
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[204:207], v9, s[28:31], 0 offen
	v_mfma_f32_16x16x32_f16 v[134:137], a[76:79], a[4:7], v[24:27]
	v_cvt_pk_f16_f32 v162, v190, v191
	v_mov_b32_e32 v204, v208
	v_mov_b32_e32 v205, v209
	v_mfma_f32_16x16x32_f16 v[130:133], a[68:71], a[4:7], v[130:133]
	v_cvt_pk_f16_f32 v163, v192, v193
	v_cvt_pk_f16_f32 v166, v186, v187
	v_cvt_pk_f16_f32 v167, v188, v189
	v_cvt_pk_f16_f32 v170, v182, v183
	v_cvt_pk_f16_f32 v171, v184, v185
	v_cvt_pk_f16_f32 v164, v174, v175
	v_cvt_pk_f16_f32 v165, v176, v177
	v_cvt_pk_f16_f32 v168, v2, v3
	v_cvt_pk_f16_f32 v169, v4, v5
	buffer_store_dwordx4 v[202:205], v12, s[28:31], 0 offen
	v_mfma_f32_16x16x32_f16 v[2:5], a[72:75], a[8:11], v[150:153]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[162:165]
	ds_write_b128 v1, v[166:169]
	ds_write_b128 v10, v[170:173]
	ds_write_b128 v11, v[178:181]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[142:145], v6
	ds_read_b128 v[148:151], v6 offset:128
	ds_read_b128 v[164:167], v6 offset:256
	ds_read_b128 v[168:171], v6 offset:384
	v_mfma_f32_16x16x32_f16 v[158:161], a[88:91], a[8:11], v[158:161]
	s_lshl_b32 s4, s9, 14
	s_lshl_b32 s5, s9, 9
	s_and_b32 s25, s17, 0xffff
	v_mfma_f32_16x16x32_f16 v[18:21], a[88:91], a[0:3], v[18:21]
	s_mov_b32 s26, s6
	s_mov_b32 s27, s7
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v162, v142
	v_mfma_f32_16x16x32_f16 v[154:157], a[80:83], a[8:11], v[154:157]
	v_mov_b32_e32 v163, v143
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v146, v166
	v_mov_b32_e32 v147, v167
	v_mfma_f32_16x16x32_f16 v[14:17], a[64:67], a[8:11], v[14:17]
	s_add_i32 s4, s4, 0
	s_and_b32 s5, s5, 0xffffe00
	buffer_store_dwordx4 v[162:165], v7, s[24:27], 0 offen
	v_mfma_f32_16x16x32_f16 v[138:141], a[80:83], a[0:3], v[138:141]
	buffer_store_dwordx4 v[144:147], v8, s[24:27], 0 offen
	v_mov_b32_e32 v166, v148
	v_mov_b32_e32 v167, v149
	v_mfma_f32_16x16x32_f16 v[134:137], a[72:75], a[0:3], v[134:137]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v152, v170
	v_mov_b32_e32 v153, v171
	v_cvt_pk_f16_f32 v142, v158, v159
	v_mfma_f32_16x16x32_f16 v[130:133], a[64:67], a[0:3], v[130:133]
	v_cvt_pk_f16_f32 v143, v160, v161
	v_cvt_pk_f16_f32 v2, v2, v3
	v_cvt_pk_f16_f32 v144, v18, v19
	v_cvt_pk_f16_f32 v145, v20, v21
	s_add_i32 s4, s4, s5
	buffer_store_dwordx4 v[166:169], v9, s[24:27], 0 offen
	buffer_store_dwordx4 v[150:153], v12, s[24:27], 0 offen
	v_cvt_pk_f16_f32 v146, v154, v155
	v_cvt_pk_f16_f32 v147, v156, v157
	v_cvt_pk_f16_f32 v3, v4, v5
	v_cvt_pk_f16_f32 v14, v14, v15
	v_cvt_pk_f16_f32 v15, v16, v17
	v_cvt_pk_f16_f32 v148, v138, v139
	v_cvt_pk_f16_f32 v149, v140, v141
	v_cvt_pk_f16_f32 v4, v134, v135
	v_cvt_pk_f16_f32 v5, v136, v137
	v_cvt_pk_f16_f32 v16, v130, v131
	v_cvt_pk_f16_f32 v17, v132, v133
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[142:145]
	ds_write_b128 v1, v[146:149]
	ds_write_b128 v10, v[2:5]
	ds_write_b128 v11, v[14:17]
	v_add_u32_e32 v2, s4, v22
	v_add_u32_e32 v13, 0x18bc0, v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[130:133], v13
	ds_read_b128 v[14:17], v13 offset:64
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[2:5], v[130:133], a[60:63], v[126:129]
	s_nop 2
	ds_read_b128 v[126:129], v13 offset:128
	ds_read_b128 v[134:137], v13 offset:192
	ds_read_b128 v[138:141], v13 offset:576
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[18:21], v[126:129], a[60:63], v[122:125]
	s_nop 2
	ds_read_b128 v[122:125], v13 offset:512
	ds_read_b128 v[150:153], v6
	ds_read_b128 v[156:159], v6 offset:128
	ds_read_b128 v[162:165], v6 offset:256
	ds_read_b128 v[166:169], v6 offset:384
	s_and_b32 s21, s11, 0xffff
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[142:145], v[122:125], a[60:63], v[118:121]
	s_nop 2
	ds_read_b128 v[118:121], v13 offset:640
	s_mov_b32 s22, s6
	s_mov_b32 s23, s7
	v_mfma_f32_16x16x32_f16 v[146:149], v[138:141], a[56:59], v[142:145]
	s_waitcnt lgkmcnt(4)
	v_mov_b32_e32 v160, v150
	v_mov_b32_e32 v161, v151
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[160:163], v7, s[20:23], 0 offen
	ds_read_b128 v[142:145], v13 offset:704
	v_mfma_f32_16x16x32_f16 v[110:113], v[130:133], a[48:51], v[110:113]
	v_mov_b32_e32 v154, v164
	v_mov_b32_e32 v155, v165
	v_mov_b32_e32 v164, v156
	v_mfma_f32_16x16x32_f16 v[106:109], v[126:129], a[48:51], v[106:109]
	v_mov_b32_e32 v165, v157
	s_waitcnt lgkmcnt(2)
	v_mov_b32_e32 v160, v168
	v_mov_b32_e32 v161, v169
	v_mfma_f32_16x16x32_f16 v[102:105], v[122:125], a[48:51], v[102:105]
	buffer_store_dwordx4 v[152:155], v8, s[20:23], 0 offen
	buffer_store_dwordx4 v[164:167], v9, s[20:23], 0 offen
	buffer_store_dwordx4 v[158:161], v12, s[20:23], 0 offen
	v_mfma_f32_16x16x32_f16 v[2:5], v[14:17], a[56:59], v[2:5]
	v_cvt_pk_f16_f32 v146, v146, v147
	v_cvt_pk_f16_f32 v147, v148, v149
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[18:21], v[134:137], a[56:59], v[18:21]
	s_barrier
	v_accvgpr_read_b32 v22, a124
	v_accvgpr_read_b32 v23, a125
	v_mfma_f32_16x16x32_f16 v[110:113], v[14:17], a[52:55], v[110:113]
	v_cvt_pk_f16_f32 v2, v2, v3
	v_cvt_pk_f16_f32 v3, v4, v5
	v_accvgpr_read_b32 v24, a126
	v_mfma_f32_16x16x32_f16 v[106:109], v[134:137], a[52:55], v[106:109]
	v_cvt_pk_f16_f32 v18, v18, v19
	v_cvt_pk_f16_f32 v19, v20, v21
	v_accvgpr_read_b32 v25, a127
	v_mfma_f32_16x16x32_f16 v[114:117], v[118:121], a[60:63], v[114:117]
	v_cvt_pk_f16_f32 v4, v110, v111
	v_cvt_pk_f16_f32 v5, v112, v113
	s_and_b32 s17, s10, 0xffff
	v_mfma_f32_16x16x32_f16 v[102:105], v[138:141], a[52:55], v[102:105]
	v_cvt_pk_f16_f32 v20, v106, v107
	v_cvt_pk_f16_f32 v21, v108, v109
	s_mov_b32 s18, s6
	v_mfma_f32_16x16x32_f16 v[98:101], v[118:121], a[48:51], v[98:101]
	s_mov_b32 s19, s7
	s_and_b32 s13, s3, 0xffff
	s_mov_b32 s14, s6
	v_mfma_f32_16x16x32_f16 v[114:117], v[142:145], a[56:59], v[114:117]
	v_cvt_pk_f16_f32 v148, v102, v103
	v_cvt_pk_f16_f32 v149, v104, v105
	ds_write_b128 v0, v[2:5]
	ds_write_b128 v1, v[18:21]
	ds_write_b128 v10, v[146:149]
	v_mfma_f32_16x16x32_f16 v[98:101], v[142:145], a[52:55], v[98:101]
	s_mov_b32 s15, s7
	s_and_b32 s9, s2, 0xffff
	v_cvt_pk_f16_f32 v114, v114, v115
	v_mfma_f32_16x16x32_f16 v[86:89], v[122:125], a[44:47], v[86:89]
	v_cvt_pk_f16_f32 v115, v116, v117
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	v_mfma_f32_16x16x32_f16 v[82:85], v[118:121], a[44:47], v[82:85]
	v_cvt_pk_f16_f32 v116, v98, v99
	v_cvt_pk_f16_f32 v117, v100, v101
	ds_write_b128 v11, v[114:117]
	v_mfma_f32_16x16x32_f16 v[2:5], v[122:125], a[36:39], v[70:73]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[98:101], v6
	ds_read_b128 v[104:107], v6 offset:128
	ds_read_b128 v[70:73], v6 offset:256
	ds_read_b128 v[110:113], v6 offset:384
	v_mfma_f32_16x16x32_f16 v[18:21], v[118:121], a[36:39], v[66:69]
	s_and_b32 s1, s1, 0xffff
	s_waitcnt lgkmcnt(2)
	v_mov_b32_e32 v108, v104
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v102, v72
	v_mfma_f32_16x16x32_f16 v[94:97], v[130:133], a[44:47], v[94:97]
	v_mov_b32_e32 v68, v98
	v_mov_b32_e32 v69, v99
	v_mov_b32_e32 v103, v73
	v_mfma_f32_16x16x32_f16 v[78:81], v[130:133], a[36:39], v[78:81]
	v_mov_b32_e32 v109, v105
	buffer_store_dwordx4 v[68:71], v7, s[16:19], 0 offen
	buffer_store_dwordx4 v[100:103], v8, s[16:19], 0 offen
	v_mfma_f32_16x16x32_f16 v[74:77], v[126:129], a[36:39], v[74:77]
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[108:111], v9, s[16:19], 0 offen
	s_mov_b32 s2, s6
	s_mov_b32 s3, s7
	v_mfma_f32_16x16x32_f16 v[90:93], v[126:129], a[44:47], v[90:93]
	v_mov_b32_e32 v108, v112
	v_mov_b32_e32 v109, v113
	buffer_store_dwordx4 v[106:109], v12, s[16:19], 0 offen
	v_mfma_f32_16x16x32_f16 v[86:89], v[138:141], a[40:43], v[86:89]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[82:85], v[142:145], a[40:43], v[82:85]
	v_mfma_f32_16x16x32_f16 v[2:5], v[138:141], a[32:35], v[2:5]
	s_nop 3
	v_cvt_pk_f16_f32 v86, v86, v87
	v_cvt_pk_f16_f32 v87, v88, v89
	v_mfma_f32_16x16x32_f16 v[18:21], v[142:145], a[32:35], v[18:21]
	v_cvt_pk_f16_f32 v82, v82, v83
	v_cvt_pk_f16_f32 v83, v84, v85
	v_mfma_f32_16x16x32_f16 v[94:97], v[14:17], a[40:43], v[94:97]
	v_cvt_pk_f16_f32 v88, v2, v3
	v_cvt_pk_f16_f32 v89, v4, v5
	v_mfma_f32_16x16x32_f16 v[78:81], v[14:17], a[32:35], v[78:81]
	s_nop 1
	v_cvt_pk_f16_f32 v84, v18, v19
	v_cvt_pk_f16_f32 v85, v20, v21
	v_mfma_f32_16x16x32_f16 v[74:77], v[134:137], a[32:35], v[74:77]
	v_cvt_pk_f16_f32 v66, v94, v95
	v_cvt_pk_f16_f32 v67, v96, v97
	v_mfma_f32_16x16x32_f16 v[90:93], v[134:137], a[40:43], v[90:93]
	v_cvt_pk_f16_f32 v68, v78, v79
	v_cvt_pk_f16_f32 v69, v80, v81
	v_mfma_f32_16x16x32_f16 v[62:65], v[130:133], a[28:31], v[62:65]
	s_nop 1
	v_cvt_pk_f16_f32 v72, v74, v75
	v_cvt_pk_f16_f32 v73, v76, v77
	v_mfma_f32_16x16x32_f16 v[54:57], v[122:125], a[28:31], v[54:57]
	v_cvt_pk_f16_f32 v70, v90, v91
	v_cvt_pk_f16_f32 v71, v92, v93
	ds_write_b128 v0, v[66:69]
	ds_write_b128 v1, v[70:73]
	ds_write_b128 v10, v[86:89]
	v_mfma_f32_16x16x32_f16 v[50:53], v[118:121], a[28:31], v[50:53]
	ds_write_b128 v11, v[82:85]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[2:5], v[130:133], a[20:23], v[28:31]
	v_mfma_f32_16x16x32_f16 v[18:21], v[126:129], a[20:23], v[42:45]
	s_nop 2
	ds_read_b128 v[42:45], v6
	ds_read_b128 v[66:69], v6 offset:128
	ds_read_b128 v[72:75], v6 offset:256
	ds_read_b128 v[76:79], v6 offset:384
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v70, v42
	v_mfma_f32_16x16x32_f16 v[34:37], v[118:121], a[20:23], v[46:49]
	v_mov_b32_e32 v71, v43
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[70:73], v7, s[12:15], 0 offen
	v_mfma_f32_16x16x32_f16 v[58:61], v[126:129], a[28:31], v[58:61]
	v_mov_b32_e32 v46, v74
	v_mov_b32_e32 v47, v75
	buffer_store_dwordx4 v[44:47], v8, s[12:15], 0 offen
	v_mfma_f32_16x16x32_f16 v[38:41], v[122:125], a[20:23], v[38:41]
	v_mov_b32_e32 v74, v66
	v_mov_b32_e32 v75, v67
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v70, v78
	v_mfma_f32_16x16x32_f16 v[30:33], v[130:133], a[12:15], v[22:25]
	v_mov_b32_e32 v71, v79
	buffer_store_dwordx4 v[74:77], v9, s[12:15], 0 offen
	buffer_store_dwordx4 v[68:71], v12, s[12:15], 0 offen
	v_accvgpr_read_b32 v22, a120
	v_accvgpr_read_b32 v23, a121
	v_accvgpr_read_b32 v24, a122
	v_accvgpr_read_b32 v25, a123
	v_mfma_f32_16x16x32_f16 v[62:65], v[14:17], a[24:27], v[62:65]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[26:29], v[126:129], a[12:15], v[22:25]
	s_nop 2
	v_accvgpr_read_b32 v22, a116
	v_mfma_f32_16x16x32_f16 v[54:57], v[138:141], a[24:27], v[54:57]
	v_accvgpr_read_b32 v23, a117
	v_accvgpr_read_b32 v24, a118
	v_accvgpr_read_b32 v25, a119
	v_mfma_f32_16x16x32_f16 v[50:53], v[142:145], a[24:27], v[50:53]
	v_cvt_pk_f16_f32 v42, v62, v63
	v_cvt_pk_f16_f32 v43, v64, v65
	v_mfma_f32_16x16x32_f16 v[2:5], v[14:17], a[16:19], v[2:5]
	s_nop 0
	v_cvt_pk_f16_f32 v54, v54, v55
	v_cvt_pk_f16_f32 v55, v56, v57
	v_mfma_f32_16x16x32_f16 v[18:21], v[134:137], a[16:19], v[18:21]
	s_nop 0
	v_cvt_pk_f16_f32 v50, v50, v51
	v_cvt_pk_f16_f32 v51, v52, v53
	v_mfma_f32_16x16x32_f16 v[34:37], v[142:145], a[16:19], v[34:37]
	v_cvt_pk_f16_f32 v44, v2, v3
	v_cvt_pk_f16_f32 v45, v4, v5
	v_mfma_f32_16x16x32_f16 v[58:61], v[134:137], a[24:27], v[58:61]
	s_nop 0
	v_cvt_pk_f16_f32 v48, v18, v19
	v_cvt_pk_f16_f32 v49, v20, v21
	v_accvgpr_read_b32 v18, a112
	v_mfma_f32_16x16x32_f16 v[38:41], v[138:141], a[16:19], v[38:41]
	v_cvt_pk_f16_f32 v52, v34, v35
	v_cvt_pk_f16_f32 v53, v36, v37
	v_accvgpr_read_b32 v19, a113
	v_mfma_f32_16x16x32_f16 v[22:25], v[122:125], a[12:15], v[22:25]
	v_cvt_pk_f16_f32 v46, v58, v59
	v_cvt_pk_f16_f32 v47, v60, v61
	v_accvgpr_read_b32 v20, a114
	v_mfma_f32_16x16x32_f16 v[2:5], v[138:141], a[8:11], v[22:25]
	v_cvt_pk_f16_f32 v56, v38, v39
	v_cvt_pk_f16_f32 v57, v40, v41
	ds_write_b128 v0, v[42:45]
	ds_write_b128 v1, v[46:49]
	ds_write_b128 v10, v[54:57]
	v_accvgpr_read_b32 v22, a108
	ds_write_b128 v11, v[50:53]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[36:39], v6
	ds_read_b128 v[42:45], v6 offset:128
	ds_read_b128 v[48:51], v6 offset:256
	ds_read_b128 v[52:55], v6 offset:384
	v_accvgpr_read_b32 v23, a109
	v_accvgpr_read_b32 v24, a110
	v_accvgpr_read_b32 v25, a111
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v46, v36
	v_mov_b32_e32 v47, v37
	v_mfma_f32_16x16x32_f16 v[22:25], v[130:133], a[4:7], v[22:25]
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v40, v50
	v_mov_b32_e32 v41, v51
	buffer_store_dwordx4 v[46:49], v7, s[8:11], 0 offen
	buffer_store_dwordx4 v[38:41], v8, s[8:11], 0 offen
	v_mfma_f32_16x16x32_f16 v[30:33], v[14:17], a[8:11], v[30:33]
	v_accvgpr_read_b32 v21, a115
	v_accvgpr_read_b32 v38, a96
	v_accvgpr_read_b32 v34, a100
	v_mfma_f32_16x16x32_f16 v[14:17], v[14:17], a[0:3], v[22:25]
	v_accvgpr_read_b32 v39, a97
	v_accvgpr_read_b32 v40, a98
	v_accvgpr_read_b32 v41, a99
	v_accvgpr_read_b32 v22, a104
	v_mfma_f32_16x16x32_f16 v[18:21], v[118:121], a[12:15], v[18:21]
	v_accvgpr_read_b32 v23, a105
	v_accvgpr_read_b32 v24, a106
	v_accvgpr_read_b32 v25, a107
	v_accvgpr_read_b32 v35, a101
	v_accvgpr_read_b32 v36, a102
	v_accvgpr_read_b32 v37, a103
	v_mfma_f32_16x16x32_f16 v[38:41], v[118:121], a[4:7], v[38:41]
	v_mov_b32_e32 v50, v42
	v_mov_b32_e32 v51, v43
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v46, v54
	v_mfma_f32_16x16x32_f16 v[22:25], v[126:129], a[4:7], v[22:25]
	v_mov_b32_e32 v47, v55
	v_cvt_pk_f16_f32 v30, v30, v31
	v_cvt_pk_f16_f32 v31, v32, v33
	v_mfma_f32_16x16x32_f16 v[34:37], v[122:125], a[4:7], v[34:37]
	v_cvt_pk_f16_f32 v2, v2, v3
	v_cvt_pk_f16_f32 v3, v4, v5
	v_cvt_pk_f16_f32 v32, v14, v15
	v_mfma_f32_16x16x32_f16 v[26:29], v[134:137], a[8:11], v[26:29]
	v_cvt_pk_f16_f32 v33, v16, v17
	buffer_store_dwordx4 v[50:53], v9, s[8:11], 0 offen
	buffer_store_dwordx4 v[44:47], v12, s[8:11], 0 offen
	v_mfma_f32_16x16x32_f16 v[18:21], v[142:145], a[8:11], v[18:21]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[38:41], v[142:145], a[0:3], v[38:41]
	s_nop 0
	v_cvt_pk_f16_f32 v26, v26, v27
	v_cvt_pk_f16_f32 v27, v28, v29
	v_mfma_f32_16x16x32_f16 v[22:25], v[134:137], a[0:3], v[22:25]
	s_nop 0
	v_cvt_pk_f16_f32 v18, v18, v19
	v_cvt_pk_f16_f32 v19, v20, v21
	v_mfma_f32_16x16x32_f16 v[34:37], v[138:141], a[0:3], v[34:37]
	v_cvt_pk_f16_f32 v20, v38, v39
	v_cvt_pk_f16_f32 v21, v40, v41
	s_nop 1
	v_cvt_pk_f16_f32 v28, v22, v23
	v_cvt_pk_f16_f32 v29, v24, v25
	s_nop 1
	v_cvt_pk_f16_f32 v4, v34, v35
	v_cvt_pk_f16_f32 v5, v36, v37
	ds_write_b128 v0, v[30:33]
	ds_write_b128 v1, v[26:29]
	ds_write_b128 v10, v[2:5]
	ds_write_b128 v11, v[18:21]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[0:3], v6
	ds_read_b128 v[14:17], v6 offset:128
	ds_read_b128 v[20:23], v6 offset:256
	ds_read_b128 v[24:27], v6 offset:384
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v1
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[18:21], v7, s[0:3], 0 offen
	v_mov_b32_e32 v4, v22
	v_mov_b32_e32 v5, v23
	v_mov_b32_e32 v22, v14
	v_mov_b32_e32 v23, v15
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v26
	v_mov_b32_e32 v19, v27
	buffer_store_dwordx4 v[2:5], v8, s[0:3], 0 offen
	buffer_store_dwordx4 v[22:25], v9, s[0:3], 0 offen
	buffer_store_dwordx4 v[16:19], v12, s[0:3], 0 offen
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
		.amdhsa_next_free_sgpr 60
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
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
	.set v8_beyond_hotloop.num_agpr, 256
	.set v8_beyond_hotloop.numbered_sgpr, 60
	.set v8_beyond_hotloop.num_named_barrier, 0
	.set v8_beyond_hotloop.private_seg_size, 0
	.set v8_beyond_hotloop.uses_vcc, 1
	.set v8_beyond_hotloop.uses_flat_scratch, 0
	.set v8_beyond_hotloop.has_dyn_sized_stack, 0
	.set v8_beyond_hotloop.has_recursion, 0
	.set v8_beyond_hotloop.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15940
; TotalNumSgprs: 66
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 8
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 512
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
	.byte	1                               ; Abbrev [1] 0xb:0x71 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x4b DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x2d DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	111                             ; DW_AT_call_line
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
	.byte	5                               ; Abbrev [5] 0x6e:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	206                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
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
.Ldebug_ranges2:
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
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
      - .offset:         44
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
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         v8_beyond_hotloop.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
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
