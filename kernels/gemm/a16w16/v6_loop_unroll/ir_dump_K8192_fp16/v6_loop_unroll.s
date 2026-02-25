	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v6_loop_unroll                  ; -- Begin function v6_loop_unroll
	.p2align	8
	.type	v6_loop_unroll,@function
v6_loop_unroll:                         ; @v6_loop_unroll
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.3:
	.file	1 "/var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v6_loop_unroll" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.4:
.LBB0_0:
	.file	2 "/var/lib/jenkins/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s9, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 8
	s_abs_i32 s1, s0
	v_mov_b32_e32 v18, v0
	v_cvt_f32_u32_e32 v0, s1
	s_sub_i32 s17, 0, s1
	s_abs_i32 s14, s16
	v_readfirstlane_b32 s24, v18
	v_rcp_iflag_f32_e32 v0, v0
	s_xor_b32 s13, s16, s0
	s_bfe_u32 s15, s24, 0x20006
	s_ashr_i32 s13, s13, 31
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	v_lshlrev_b32_e32 v8, 3, v18
	scratch_store_dword off, v8, off offset:4 ; 4-byte Folded Spill
	v_and_b32_e32 v8, 56, v8
	v_readfirstlane_b32 s18, v0
	s_mul_i32 s17, s17, s18
	s_mul_hi_u32 s17, s18, s17
	s_add_i32 s18, s18, s17
	s_mul_hi_u32 s17, s14, s18
	s_mul_i32 s18, s17, s1
	s_sub_i32 s14, s14, s18
	s_add_i32 s18, s17, 1
	s_sub_i32 s19, s14, s1
	s_cmp_ge_u32 s14, s1
	s_cselect_b32 s17, s18, s17
	s_cselect_b32 s14, s19, s14
	s_add_i32 s18, s17, 1
	s_cmp_ge_u32 s14, s1
	s_cselect_b32 s1, s18, s17
	s_xor_b32 s1, s1, s13
	s_sub_i32 s1, s1, s13
	s_mul_i32 s0, s1, s0
	s_lshl_b32 s13, s1, 8
	s_sub_i32 s14, s16, s0
	s_mul_i32 s0, s13, s10
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s2, s0
	s_addc_u32 s55, s3, s1
	s_lshl_b32 s14, s14, 8
	v_and_b32_e32 v0, 63, v18
	s_mul_i32 s2, s14, s11
	v_lshl_or_b32 v17, s15, 6, v0
	v_lshlrev_b32_e32 v0, 1, v18
	s_ashr_i32 s3, s2, 31
	v_and_b32_e32 v0, 0x70, v0
	s_lshl_b64 s[2:3], s[2:3], 1
	v_or_b32_e32 v0, s15, v0
	s_add_u32 s16, s4, s2
	s_mul_i32 s4, s15, 0x420
	v_or_b32_e32 v1, 4, v0
	v_or_b32_e32 v2, 8, v0
	v_or_b32_e32 v3, 12, v0
	v_or_b32_e32 v4, 0x80, v0
	v_or_b32_e32 v5, 0x84, v0
	v_or_b32_e32 v6, 0x88, v0
	v_or_b32_e32 v7, 0x8c, v0
	s_addc_u32 s56, s5, s3
	v_mul_lo_u32 v9, v0, s10
	s_add_i32 s5, s4, 0
	v_mul_lo_u32 v10, v1, s10
	v_mul_lo_u32 v11, v2, s10
	v_mul_lo_u32 v12, v3, s10
	v_mul_lo_u32 v13, v4, s10
	v_mul_lo_u32 v14, v5, s10
	v_mul_lo_u32 v15, v6, s10
	v_mul_lo_u32 v16, v7, s10
	s_and_b32 s1, s55, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_add_lshl_u32 v9, v9, v8, 1
	s_mov_b32 m0, s5
	s_add_i32 s10, s5, 0x1080
	v_mul_lo_u32 v0, v0, s11
	v_mul_lo_u32 v1, v1, s11
	v_mul_lo_u32 v2, v2, s11
	v_mul_lo_u32 v3, v3, s11
	v_mul_lo_u32 v4, v4, s11
	v_mul_lo_u32 v5, v5, s11
	v_mul_lo_u32 v6, v6, s11
	v_mul_lo_u32 v7, v7, s11
	buffer_load_dwordx4 v9, s[0:3], 0 offen lds
	v_add_lshl_u32 v10, v10, v8, 1
	s_mov_b32 m0, s10
	s_add_i32 s11, s5, 0x2100
	buffer_load_dwordx4 v10, s[0:3], 0 offen lds
	v_add_lshl_u32 v11, v11, v8, 1
	s_mov_b32 m0, s11
	s_add_i32 s25, s5, 0x3180
	buffer_load_dwordx4 v11, s[0:3], 0 offen lds
	v_add_lshl_u32 v12, v12, v8, 1
	s_mov_b32 m0, s25
	s_add_i32 s26, s5, 0x4200
	buffer_load_dwordx4 v12, s[0:3], 0 offen lds
	v_add_lshl_u32 v13, v13, v8, 1
	s_mov_b32 m0, s26
	s_add_i32 s27, s5, 0x5280
	buffer_load_dwordx4 v13, s[0:3], 0 offen lds
	v_add_lshl_u32 v14, v14, v8, 1
	s_mov_b32 m0, s27
	s_add_i32 s28, s5, 0x6300
	buffer_load_dwordx4 v14, s[0:3], 0 offen lds
	v_add_lshl_u32 v15, v15, v8, 1
	s_mov_b32 m0, s28
	s_add_i32 s29, s5, 0x7380
	buffer_load_dwordx4 v15, s[0:3], 0 offen lds
	v_add_lshl_u32 v16, v16, v8, 1
	s_mov_b32 m0, s29
	s_add_i32 s52, s4, 0x1080
	buffer_load_dwordx4 v16, s[0:3], 0 offen lds
	s_add_i32 s1, 0, 0x107e0
	s_add_i32 s30, s1, s4
	s_add_i32 s53, s4, 0x2100
	s_and_b32 s17, s56, 0xffff
	s_mov_b32 s18, s2
	s_mov_b32 s19, s3
	v_add_lshl_u32 v0, v0, v8, 1
	s_mov_b32 m0, s30
	s_add_i32 s31, s1, s52
	s_add_i32 s54, s4, 0x3180
	buffer_load_dwordx4 v0, s[16:19], 0 offen lds
	v_add_lshl_u32 v1, v1, v8, 1
	s_mov_b32 m0, s31
	s_add_i32 s33, s1, s53
	s_add_i32 s57, s4, 0x4200
	buffer_load_dwordx4 v1, s[16:19], 0 offen lds
	v_add_lshl_u32 v2, v2, v8, 1
	s_mov_b32 m0, s33
	s_add_i32 s34, s1, s54
	s_add_i32 s58, s4, 0x5280
	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
	v_add_lshl_u32 v3, v3, v8, 1
	s_mov_b32 m0, s34
	s_add_i32 s35, s1, s57
	s_add_i32 s59, s4, 0x6300
	buffer_load_dwordx4 v3, s[16:19], 0 offen lds
	v_add_lshl_u32 v4, v4, v8, 1
	s_mov_b32 m0, s35
	s_add_i32 s36, s1, s58
	s_add_i32 s60, s4, 0x7380
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	v_add_lshl_u32 v5, v5, v8, 1
	s_mov_b32 m0, s36
	s_add_i32 s37, s1, s59
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	v_add_lshl_u32 v6, v6, v8, 1
	s_mov_b32 m0, s37
	s_add_i32 s38, s1, s60
	buffer_load_dwordx4 v6, s[16:19], 0 offen lds
	v_add_lshl_u32 v7, v7, v8, 1
	s_mov_b32 m0, s38
	s_add_u32 s48, s0, 0x80
	buffer_load_dwordx4 v7, s[16:19], 0 offen lds
	s_addc_u32 s17, s55, 0
	s_add_u32 s20, s16, 0x80
	s_addc_u32 s18, s56, 0
	s_add_i32 s39, s30, 0xffff7c20
	s_and_b32 s49, s17, 0xffff
	s_mov_b32 s50, s2
	s_mov_b32 s51, s3
	s_mov_b32 m0, s39
	s_add_i32 s40, s30, 0xffff8ca0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v9, s[48:51], 0 offen lds
	s_mov_b32 m0, s40
	s_add_i32 s41, s30, 0xffff9d20
	buffer_load_dwordx4 v10, s[48:51], 0 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s42, s30, 0xffffada0
	buffer_load_dwordx4 v11, s[48:51], 0 offen lds
	s_mov_b32 m0, s42
	s_add_i32 s43, s30, 0xffffbe20
	buffer_load_dwordx4 v12, s[48:51], 0 offen lds
	s_mov_b32 m0, s43
	s_add_i32 s44, s30, 0xffffcea0
	buffer_load_dwordx4 v13, s[48:51], 0 offen lds
	s_mov_b32 m0, s44
	s_add_i32 s45, s30, 0xffffdf20
	buffer_load_dwordx4 v14, s[48:51], 0 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s46, s30, 0xffffefa0
	s_add_i32 s17, 0, 0x18be0
	buffer_load_dwordx4 v15, s[48:51], 0 offen lds
	s_mov_b32 m0, s46
	s_add_i32 s47, s17, s4
	buffer_load_dwordx4 v16, s[48:51], 0 offen lds
	s_and_b32 s21, s18, 0xffff
	s_mov_b32 s22, s2
	s_mov_b32 s23, s3
	s_mov_b32 m0, s47
	s_add_i32 s48, s17, s52
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	s_mov_b32 m0, s48
	s_add_i32 s49, s17, s53
	buffer_load_dwordx4 v1, s[20:23], 0 offen lds
	s_mov_b32 m0, s49
	s_add_i32 s50, s17, s54
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_mov_b32 m0, s50
	s_add_i32 s51, s17, s57
	buffer_load_dwordx4 v3, s[20:23], 0 offen lds
	s_mov_b32 m0, s51
	s_add_i32 s52, s17, s58
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	s_mov_b32 m0, s52
	s_add_i32 s53, s17, s59
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_mov_b32 m0, s53
	s_add_i32 s54, s17, s60
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	s_mov_b32 m0, s54
	v_accvgpr_write_b32 a140, v0
	buffer_load_dwordx4 v7, s[20:23], 0 offen lds
	v_accvgpr_write_b32 a142, v2
	v_lshlrev_b32_e32 v0, 10, v18
	s_movk_i32 s4, 0x3cb0
	v_lshlrev_b32_e32 v2, 5, v18
	s_movk_i32 s17, 0x3c30
	v_accvgpr_write_b32 a141, v1
	v_bitop3_b32 v1, v17, s4, v0 bitop3:0xc8
	v_and_b32_e32 v2, 0x1e0, v2
	v_bitop3_b32 v0, v0, s17, v18 bitop3:0xc8
	v_add_u32_e32 v1, v1, v2
	s_and_b32 s4, s24, 64
	v_add_u32_e32 v0, v0, v2
	v_add_u32_e32 v1, 0, v1
	v_lshl_add_u32 v0, s4, 1, v0
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 a[120:123], v1
	ds_read_b128 a[124:127], v1 offset:64
	ds_read_b128 a[112:115], v1 offset:256
	ds_read_b128 a[116:119], v1 offset:320
	ds_read_b128 a[104:107], v1 offset:512
	ds_read_b128 a[108:111], v1 offset:576
	ds_read_b128 a[96:99], v1 offset:768
	ds_read_b128 a[100:103], v1 offset:832
	ds_read_b128 a[88:91], v1 offset:16896
	ds_read_b128 a[92:95], v1 offset:16960
	ds_read_b128 a[80:83], v1 offset:17152
	ds_read_b128 a[84:87], v1 offset:17216
	ds_read_b128 a[8:11], v1 offset:17408
	ds_read_b128 a[12:15], v1 offset:17472
	ds_read_b128 a[0:3], v1 offset:17664
	v_accvgpr_write_b32 a135, v1
	ds_read_b128 a[4:7], v1 offset:17728
	v_add_u32_e32 v1, s1, v0
	ds_read_b128 a[16:19], v1
	ds_read_b128 a[20:23], v1 offset:64
	ds_read_b128 a[24:27], v1 offset:256
	ds_read_b128 a[28:31], v1 offset:320
	ds_read_b128 a[32:35], v1 offset:512
	ds_read_b128 a[36:39], v1 offset:576
	ds_read_b128 a[40:43], v1 offset:768
	ds_read_b128 a[44:47], v1 offset:832
	ds_read_b128 a[48:51], v1 offset:16896
	ds_read_b128 a[52:55], v1 offset:16960
	ds_read_b128 a[56:59], v1 offset:17152
	ds_read_b128 a[60:63], v1 offset:17216
	ds_read_b128 a[64:67], v1 offset:17408
	ds_read_b128 a[68:71], v1 offset:17472
	ds_read_b128 a[72:75], v1 offset:17664
	ds_read_b128 a[76:79], v1 offset:17728
	s_add_u32 s20, s16, 0x180
	v_mov_b32_e32 v2, 0
	s_addc_u32 s21, s56, 0
	v_add_u32_e32 v0, 0, v0
	v_accvgpr_write_b32 a151, v2
	s_add_u32 s22, s0, 0x180
	v_accvgpr_write_b32 a148, v2
	v_accvgpr_write_b32 a149, v2
	v_accvgpr_write_b32 a150, v2
	v_accvgpr_mov_b32 a155, a151
	v_accvgpr_mov_b32 a159, a151
	v_add_u32_e32 v1, 0x18be0, v0
	v_mov_b32_e32 v118, v0
	v_add_u32_e32 v0, 0x107e0, v0
	v_accvgpr_write_b32 a131, v9
	v_accvgpr_write_b32 a132, v10
	v_accvgpr_write_b32 a133, v11
	v_accvgpr_write_b32 a134, v12
	v_accvgpr_write_b32 a136, v13
	v_accvgpr_write_b32 a137, v14
	v_accvgpr_write_b32 a138, v15
	v_accvgpr_write_b32 a139, v16
	v_accvgpr_write_b32 a143, v3
	v_accvgpr_write_b32 a144, v4
	v_accvgpr_write_b32 a145, v5
	v_accvgpr_write_b32 a146, v6
	v_accvgpr_write_b32 a147, v7
	scratch_store_dword off, v17, off       ; 4-byte Folded Spill
	scratch_store_dword off, v18, off offset:8 ; 4-byte Folded Spill
	s_addc_u32 s23, s55, 0
	s_mov_b32 s55, -2
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v22, v2
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v26, v2
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v54, v2
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v62, v2
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
	v_mov_b32_e32 v66, v2
	v_mov_b32_e32 v67, v2
	v_mov_b32_e32 v68, v2
	v_mov_b32_e32 v69, v2
	v_mov_b32_e32 v70, v2
	v_mov_b32_e32 v71, v2
	v_mov_b32_e32 v72, v2
	v_mov_b32_e32 v73, v2
	v_mov_b32_e32 v74, v2
	v_mov_b32_e32 v75, v2
	v_mov_b32_e32 v76, v2
	v_mov_b32_e32 v77, v2
	v_mov_b32_e32 v78, v2
	v_mov_b32_e32 v79, v2
	v_mov_b32_e32 v80, v2
	v_mov_b32_e32 v81, v2
	v_mov_b32_e32 v82, v2
	v_mov_b32_e32 v83, v2
	v_mov_b32_e32 v84, v2
	v_mov_b32_e32 v85, v2
	v_mov_b32_e32 v86, v2
	v_mov_b32_e32 v87, v2
	v_mov_b32_e32 v88, v2
	v_mov_b32_e32 v89, v2
	v_mov_b32_e32 v90, v2
	v_mov_b32_e32 v91, v2
	v_mov_b32_e32 v92, v2
	v_mov_b32_e32 v93, v2
	v_mov_b32_e32 v94, v2
	v_mov_b32_e32 v95, v2
	v_mov_b32_e32 v96, v2
	v_mov_b32_e32 v97, v2
	v_mov_b32_e32 v98, v2
	v_mov_b32_e32 v99, v2
	v_mov_b32_e32 v100, v2
	v_mov_b32_e32 v101, v2
	v_mov_b32_e32 v102, v2
	v_mov_b32_e32 v103, v2
	v_mov_b32_e32 v104, v2
	v_mov_b32_e32 v105, v2
	v_accvgpr_write_b32 a176, v2
	v_accvgpr_write_b32 a177, v2
	v_accvgpr_write_b32 a178, v2
	v_accvgpr_write_b32 a179, v2
	v_accvgpr_write_b32 a184, v2
	v_accvgpr_write_b32 a185, v2
	v_accvgpr_write_b32 a186, v2
	v_accvgpr_write_b32 a187, v2
	v_accvgpr_write_b32 a188, v2
	v_accvgpr_write_b32 a189, v2
	v_accvgpr_write_b32 a190, v2
	v_accvgpr_write_b32 a191, v2
	v_accvgpr_write_b32 a192, v2
	v_accvgpr_write_b32 a193, v2
	v_accvgpr_write_b32 a194, v2
	v_accvgpr_write_b32 a195, v2
	v_accvgpr_write_b32 a196, v2
	v_accvgpr_write_b32 a197, v2
	v_accvgpr_write_b32 a198, v2
	v_accvgpr_write_b32 a199, v2
	v_accvgpr_write_b32 a200, v2
	v_accvgpr_write_b32 a201, v2
	v_accvgpr_write_b32 a202, v2
	v_accvgpr_write_b32 a203, v2
	v_accvgpr_write_b32 a204, v2
	v_accvgpr_write_b32 a205, v2
	v_accvgpr_write_b32 a206, v2
	v_accvgpr_write_b32 a207, v2
	v_accvgpr_write_b32 a208, v2
	v_accvgpr_write_b32 a209, v2
	v_accvgpr_write_b32 a210, v2
	v_accvgpr_write_b32 a211, v2
	v_accvgpr_write_b32 a212, v2
	v_accvgpr_write_b32 a213, v2
	v_accvgpr_write_b32 a214, v2
	v_accvgpr_write_b32 a215, v2
	v_accvgpr_write_b32 a216, v2
	v_accvgpr_write_b32 a217, v2
	v_accvgpr_write_b32 a218, v2
	v_accvgpr_write_b32 a219, v2
	v_accvgpr_write_b32 a220, v2
	v_accvgpr_write_b32 a221, v2
	v_accvgpr_write_b32 a222, v2
	v_accvgpr_write_b32 a223, v2
	v_accvgpr_write_b32 a224, v2
	v_accvgpr_write_b32 a225, v2
	v_accvgpr_write_b32 a226, v2
	v_accvgpr_write_b32 a227, v2
	v_accvgpr_write_b32 a228, v2
	v_accvgpr_write_b32 a229, v2
	v_accvgpr_write_b32 a230, v2
	v_accvgpr_write_b32 a231, v2
	v_accvgpr_write_b32 a232, v2
	v_accvgpr_write_b32 a233, v2
	v_accvgpr_write_b32 a234, v2
	v_accvgpr_write_b32 a235, v2
	v_accvgpr_write_b32 a236, v2
	v_accvgpr_write_b32 a237, v2
	v_accvgpr_write_b32 a238, v2
	v_accvgpr_write_b32 a239, v2
	v_accvgpr_write_b32 a240, v2
	v_accvgpr_write_b32 a241, v2
	v_accvgpr_write_b32 a242, v2
	v_accvgpr_write_b32 a243, v2
	v_accvgpr_write_b32 a244, v2
	v_accvgpr_write_b32 a245, v2
	v_accvgpr_write_b32 a246, v2
	v_accvgpr_write_b32 a247, v2
	v_accvgpr_write_b32 a248, v2
	v_accvgpr_write_b32 a249, v2
	v_accvgpr_write_b32 a250, v2
	v_accvgpr_write_b32 a251, v2
	v_accvgpr_write_b32 a252, v2
	v_accvgpr_write_b32 a253, v2
	v_accvgpr_write_b32 a254, v2
	v_accvgpr_write_b32 a255, v2
	v_mov_b32_e32 v182, v2
	v_mov_b32_e32 v183, v2
	v_mov_b32_e32 v184, v2
	v_mov_b32_e32 v185, v2
	v_mov_b32_e32 v186, v2
	v_mov_b32_e32 v187, v2
	v_mov_b32_e32 v188, v2
	v_mov_b32_e32 v189, v2
	v_mov_b32_e32 v190, v2
	v_mov_b32_e32 v191, v2
	v_mov_b32_e32 v192, v2
	v_mov_b32_e32 v193, v2
	v_mov_b32_e32 v194, v2
	v_mov_b32_e32 v195, v2
	v_mov_b32_e32 v196, v2
	v_mov_b32_e32 v197, v2
	v_mov_b32_e32 v198, v2
	v_mov_b32_e32 v199, v2
	v_mov_b32_e32 v200, v2
	v_mov_b32_e32 v201, v2
	v_mov_b32_e32 v202, v2
	v_mov_b32_e32 v203, v2
	v_mov_b32_e32 v204, v2
	v_mov_b32_e32 v205, v2
	v_mov_b32_e32 v206, v2
	v_mov_b32_e32 v207, v2
	v_mov_b32_e32 v208, v2
	v_mov_b32_e32 v209, v2
	v_mov_b32_e32 v210, v2
	v_mov_b32_e32 v211, v2
	v_mov_b32_e32 v212, v2
	v_mov_b32_e32 v213, v2
	v_mov_b32_e32 v214, v2
	v_mov_b32_e32 v215, v2
	v_mov_b32_e32 v216, v2
	v_mov_b32_e32 v217, v2
	v_mov_b32_e32 v218, v2
	v_mov_b32_e32 v219, v2
	v_mov_b32_e32 v220, v2
	v_mov_b32_e32 v221, v2
	v_mov_b32_e32 v222, v2
	v_mov_b32_e32 v223, v2
	v_mov_b32_e32 v224, v2
	v_mov_b32_e32 v225, v2
	v_mov_b32_e32 v226, v2
	v_mov_b32_e32 v227, v2
	v_mov_b32_e32 v228, v2
	v_mov_b32_e32 v229, v2
	v_mov_b32_e32 v230, v2
	v_mov_b32_e32 v231, v2
	v_mov_b32_e32 v232, v2
	v_mov_b32_e32 v233, v2
	v_mov_b32_e32 v234, v2
	v_mov_b32_e32 v235, v2
	v_mov_b32_e32 v236, v2
	v_mov_b32_e32 v237, v2
	v_mov_b32_e32 v238, v2
	v_mov_b32_e32 v239, v2
	v_mov_b32_e32 v240, v2
	v_mov_b32_e32 v241, v2
	v_mov_b32_e32 v242, v2
	v_mov_b32_e32 v243, v2
	v_mov_b32_e32 v244, v2
	v_mov_b32_e32 v245, v2
	v_mov_b32_e32 v246, v2
	v_mov_b32_e32 v247, v2
	v_mov_b32_e32 v248, v2
	v_mov_b32_e32 v249, v2
	v_mov_b32_e32 v250, v2
	v_mov_b32_e32 v251, v2
	v_mov_b32_e32 v252, v2
	v_mov_b32_e32 v253, v2
	v_accvgpr_mov_b32 a154, a150
	v_accvgpr_mov_b32 a153, a149
	v_accvgpr_mov_b32 a152, a148
	v_accvgpr_mov_b32 a158, a150
	v_accvgpr_mov_b32 a157, a149
	v_accvgpr_mov_b32 a156, a148
	v_accvgpr_write_b32 a149, v1
	v_accvgpr_write_b32 a150, v0
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[2:5], a[16:19], a[120:123], v[2:5]
	s_add_u32 s16, s20, 0xffffff80
	s_addc_u32 s17, s21, -1
	s_add_u32 s0, s22, 0xffffff80
	v_mfma_f32_16x16x32_f16 v[0:3], a[20:23], a[124:127], v[2:5]
	s_addc_u32 s1, s23, -1
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 m0, s5
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[10:13], a[24:27], a[120:123], v[8:11]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[14:17], a[32:35], a[120:123], v[14:17]
	v_accvgpr_write_b32 a163, v3
	v_accvgpr_write_b32 a162, v2
	v_accvgpr_write_b32 a161, v1
	v_accvgpr_write_b32 a160, v0
	v_mfma_f32_16x16x32_f16 v[0:3], a[28:31], a[124:127], v[10:13]
	v_accvgpr_read_b32 v254, a133
	v_accvgpr_read_b32 v255, a134
	v_accvgpr_read_b32 v7, a136
	v_mfma_f32_16x16x32_f16 v[18:21], a[40:43], a[120:123], v[18:21]
	v_accvgpr_read_b32 v8, a137
	v_accvgpr_read_b32 v9, a138
	v_accvgpr_read_b32 v10, a139
	v_mfma_f32_16x16x32_f16 v[22:25], a[48:51], a[120:123], v[22:25]
	v_accvgpr_write_b32 a167, v3
	v_accvgpr_write_b32 a166, v2
	v_accvgpr_write_b32 a165, v1
	v_accvgpr_write_b32 a164, v0
	v_mfma_f32_16x16x32_f16 v[0:3], a[36:39], a[124:127], v[14:17]
	s_and_b32 s17, s17, 0xffff
	s_mov_b32 s18, s2
	s_mov_b32 s19, s3
	v_mfma_f32_16x16x32_f16 v[26:29], a[56:59], a[120:123], v[26:29]
	v_accvgpr_read_b32 v14, a132
	v_accvgpr_read_b32 v11, a140
	v_accvgpr_read_b32 v12, a141
	v_mfma_f32_16x16x32_f16 v[30:33], a[64:67], a[120:123], v[30:33]
	v_accvgpr_write_b32 a171, v3
	v_accvgpr_write_b32 a170, v2
	v_accvgpr_write_b32 a169, v1
	v_mfma_f32_16x16x32_f16 v[34:37], a[72:75], a[120:123], v[34:37]
	v_accvgpr_write_b32 a168, v0
	v_accvgpr_read_b32 v13, a142
	v_accvgpr_read_b32 v4, a143
	v_mfma_f32_16x16x32_f16 v[38:41], a[16:19], a[112:115], v[38:41]
	v_accvgpr_read_b32 v15, a144
	v_accvgpr_read_b32 v16, a145
	v_accvgpr_read_b32 v17, a146
	v_mfma_f32_16x16x32_f16 v[42:45], a[24:27], a[112:115], v[42:45]
	v_accvgpr_read_b32 v6, a147
	v_accvgpr_read_b32 v5, a135
	v_mfma_f32_16x16x32_f16 v[46:49], a[32:35], a[112:115], v[46:49]
	v_mfma_f32_16x16x32_f16 v[50:53], a[40:43], a[112:115], v[50:53]
	v_mfma_f32_16x16x32_f16 v[54:57], a[48:51], a[112:115], v[54:57]
	v_mfma_f32_16x16x32_f16 v[58:61], a[56:59], a[112:115], v[58:61]
	v_mfma_f32_16x16x32_f16 v[62:65], a[64:67], a[112:115], v[62:65]
	v_mfma_f32_16x16x32_f16 v[66:69], a[72:75], a[112:115], v[66:69]
	v_mfma_f32_16x16x32_f16 v[70:73], a[16:19], a[104:107], v[70:73]
	v_mfma_f32_16x16x32_f16 v[74:77], a[24:27], a[104:107], v[74:77]
	v_mfma_f32_16x16x32_f16 v[78:81], a[32:35], a[104:107], v[78:81]
	v_mfma_f32_16x16x32_f16 v[82:85], a[40:43], a[104:107], v[82:85]
	v_mfma_f32_16x16x32_f16 v[86:89], a[48:51], a[104:107], v[86:89]
	v_mfma_f32_16x16x32_f16 v[90:93], a[56:59], a[104:107], v[90:93]
	v_mfma_f32_16x16x32_f16 v[94:97], a[64:67], a[104:107], v[94:97]
	v_mfma_f32_16x16x32_f16 v[98:101], a[72:75], a[104:107], v[98:101]
	v_mfma_f32_16x16x32_f16 v[0:3], a[44:47], a[124:127], v[18:21]
	v_mfma_f32_16x16x32_f16 v[20:23], a[52:55], a[124:127], v[22:25]
	s_nop 1
	v_accvgpr_read_b32 v18, a131
	buffer_load_dwordx4 v18, s[0:3], 0 offen lds
	s_mov_b32 m0, s10
	v_mfma_f32_16x16x32_f16 v[26:29], a[60:63], a[124:127], v[26:29]
	buffer_load_dwordx4 v14, s[0:3], 0 offen lds
	s_mov_b32 m0, s11
	v_accvgpr_write_b32 a175, v3
	v_mfma_f32_16x16x32_f16 v[30:33], a[68:71], a[124:127], v[30:33]
	buffer_load_dwordx4 v254, s[0:3], 0 offen lds
	s_mov_b32 m0, s25
	v_accvgpr_write_b32 a174, v2
	v_mfma_f32_16x16x32_f16 v[34:37], a[76:79], a[124:127], v[34:37]
	buffer_load_dwordx4 v255, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	v_accvgpr_write_b32 a173, v1
	v_mfma_f32_16x16x32_f16 v[38:41], a[20:23], a[116:119], v[38:41]
	buffer_load_dwordx4 v7, s[0:3], 0 offen lds
	s_mov_b32 m0, s27
	v_accvgpr_write_b32 a172, v0
	v_mfma_f32_16x16x32_f16 v[42:45], a[28:31], a[116:119], v[42:45]
	buffer_load_dwordx4 v8, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	v_accvgpr_read_b32 v0, a149
	v_mfma_f32_16x16x32_f16 v[46:49], a[36:39], a[116:119], v[46:49]
	buffer_load_dwordx4 v9, s[0:3], 0 offen lds
	s_mov_b32 m0, s29
	v_mfma_f32_16x16x32_f16 v[50:53], a[44:47], a[116:119], v[50:53]
	buffer_load_dwordx4 v10, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_and_b32 s1, s23, 0xffff
	v_mfma_f32_16x16x32_f16 v[54:57], a[52:55], a[116:119], v[54:57]
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	s_mov_b32 m0, s31
	s_mov_b32 s0, s22
	v_mfma_f32_16x16x32_f16 v[58:61], a[60:63], a[116:119], v[58:61]
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	s_mov_b32 m0, s33
	v_mfma_f32_16x16x32_f16 v[62:65], a[68:71], a[116:119], v[62:65]
	buffer_load_dwordx4 v13, s[16:19], 0 offen lds
	s_mov_b32 m0, s34
	v_mfma_f32_16x16x32_f16 v[66:69], a[76:79], a[116:119], v[66:69]
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_mov_b32 m0, s35
	v_mfma_f32_16x16x32_f16 v[70:73], a[20:23], a[108:111], v[70:73]
	buffer_load_dwordx4 v15, s[16:19], 0 offen lds
	s_mov_b32 m0, s36
	v_mfma_f32_16x16x32_f16 v[74:77], a[28:31], a[108:111], v[74:77]
	buffer_load_dwordx4 v16, s[16:19], 0 offen lds
	s_mov_b32 m0, s37
	v_mfma_f32_16x16x32_f16 v[78:81], a[36:39], a[108:111], v[78:81]
	buffer_load_dwordx4 v17, s[16:19], 0 offen lds
	s_mov_b32 m0, s38
	v_mfma_f32_16x16x32_f16 v[82:85], a[44:47], a[108:111], v[82:85]
	buffer_load_dwordx4 v6, s[16:19], 0 offen lds
	s_mov_b32 m0, s39
	v_mfma_f32_16x16x32_f16 v[86:89], a[52:55], a[108:111], v[86:89]
	v_mfma_f32_16x16x32_f16 v[90:93], a[60:63], a[108:111], v[90:93]
	v_mfma_f32_16x16x32_f16 v[94:97], a[68:71], a[108:111], v[94:97]
	v_mfma_f32_16x16x32_f16 v[98:101], a[76:79], a[108:111], v[98:101]
	v_mfma_f32_16x16x32_f16 v[102:105], a[16:19], a[96:99], v[102:105]
	v_mfma_f32_16x16x32_f16 a[104:107], a[24:27], a[96:99], a[176:179]
	v_mfma_f32_16x16x32_f16 a[108:111], a[32:35], a[96:99], a[184:187]
	v_mfma_f32_16x16x32_f16 a[112:115], a[40:43], a[96:99], a[188:191]
	v_mfma_f32_16x16x32_f16 a[116:119], a[48:51], a[96:99], a[192:195]
	v_mfma_f32_16x16x32_f16 a[120:123], a[56:59], a[96:99], a[196:199]
	v_mfma_f32_16x16x32_f16 a[124:127], a[64:67], a[96:99], a[200:203]
	v_mfma_f32_16x16x32_f16 a[96:99], a[72:75], a[96:99], a[204:207]
	v_mfma_f32_16x16x32_f16 a[204:207], a[76:79], a[100:103], a[96:99]
	v_mfma_f32_16x16x32_f16 a[96:99], a[24:27], a[88:91], a[212:215]
	v_mfma_f32_16x16x32_f16 a[212:215], a[28:31], a[92:95], a[96:99]
	v_mfma_f32_16x16x32_f16 a[96:99], a[32:35], a[88:91], a[216:219]
	v_mfma_f32_16x16x32_f16 a[216:219], a[36:39], a[92:95], a[96:99]
	v_mfma_f32_16x16x32_f16 a[96:99], a[40:43], a[88:91], a[220:223]
	v_mfma_f32_16x16x32_f16 a[220:223], a[44:47], a[92:95], a[96:99]
	v_mfma_f32_16x16x32_f16 a[96:99], a[48:51], a[88:91], a[224:227]
	v_mfma_f32_16x16x32_f16 a[224:227], a[52:55], a[92:95], a[96:99]
	v_mfma_f32_16x16x32_f16 a[96:99], a[56:59], a[88:91], a[228:231]
	v_mfma_f32_16x16x32_f16 v[102:105], a[20:23], a[100:103], v[102:105]
	v_mfma_f32_16x16x32_f16 a[180:183], a[28:31], a[100:103], a[104:107]
	v_mfma_f32_16x16x32_f16 a[184:187], a[36:39], a[100:103], a[108:111]
	v_mfma_f32_16x16x32_f16 a[188:191], a[44:47], a[100:103], a[112:115]
	v_mfma_f32_16x16x32_f16 a[192:195], a[52:55], a[100:103], a[116:119]
	v_mfma_f32_16x16x32_f16 a[196:199], a[60:63], a[100:103], a[120:123]
	v_mfma_f32_16x16x32_f16 a[200:203], a[68:71], a[100:103], a[124:127]
	v_mfma_f32_16x16x32_f16 a[100:103], a[16:19], a[88:91], a[208:211]
	v_mfma_f32_16x16x32_f16 a[228:231], a[60:63], a[92:95], a[96:99]
	v_mfma_f32_16x16x32_f16 a[96:99], a[64:67], a[88:91], a[232:235]
	v_mfma_f32_16x16x32_f16 a[88:91], a[72:75], a[88:91], a[236:239]
	v_mfma_f32_16x16x32_f16 a[236:239], a[76:79], a[92:95], a[88:91]
	v_mfma_f32_16x16x32_f16 a[88:91], a[24:27], a[80:83], a[244:247]
	v_mfma_f32_16x16x32_f16 v[198:201], a[16:19], a[8:11], v[198:201]
	v_mfma_f32_16x16x32_f16 v[202:205], a[24:27], a[8:11], v[202:205]
	v_mfma_f32_16x16x32_f16 v[206:209], a[32:35], a[8:11], v[206:209]
	v_mfma_f32_16x16x32_f16 v[210:213], a[40:43], a[8:11], v[210:213]
	v_mfma_f32_16x16x32_f16 v[214:217], a[48:51], a[8:11], v[214:217]
	v_mfma_f32_16x16x32_f16 v[218:221], a[56:59], a[8:11], v[218:221]
	v_mfma_f32_16x16x32_f16 v[222:225], a[64:67], a[8:11], v[222:225]
	v_mfma_f32_16x16x32_f16 v[226:229], a[72:75], a[8:11], v[226:229]
	v_accvgpr_mov_b32 a8, a152
	v_accvgpr_mov_b32 a9, a153
	v_accvgpr_mov_b32 a10, a154
	v_accvgpr_mov_b32 a11, a155
	v_mfma_f32_16x16x32_f16 a[244:247], a[28:31], a[84:87], a[88:91]
	s_nop 0
	v_mfma_f32_16x16x32_f16 a[8:11], a[64:67], a[0:3], a[8:11]
	v_mfma_f32_16x16x32_f16 a[88:91], a[32:35], a[80:83], a[248:251]
	v_mfma_f32_16x16x32_f16 a[152:155], a[68:71], a[4:7], a[8:11]
	s_nop 5
	v_accvgpr_mov_b32 a8, a156
	v_accvgpr_mov_b32 a9, a157
	v_accvgpr_mov_b32 a10, a158
	v_accvgpr_mov_b32 a11, a159
	v_mfma_f32_16x16x32_f16 a[176:179], a[16:19], a[80:83], a[240:243]
	v_mfma_f32_16x16x32_f16 a[248:251], a[36:39], a[84:87], a[88:91]
	v_mfma_f32_16x16x32_f16 a[88:91], a[40:43], a[80:83], a[252:255]
	v_mfma_f32_16x16x32_f16 v[182:185], a[48:51], a[80:83], v[182:185]
	v_mfma_f32_16x16x32_f16 v[186:189], a[56:59], a[80:83], v[186:189]
	v_mfma_f32_16x16x32_f16 v[190:193], a[64:67], a[80:83], v[190:193]
	v_mfma_f32_16x16x32_f16 v[194:197], a[72:75], a[80:83], v[194:197]
	v_mfma_f32_16x16x32_f16 v[230:233], a[16:19], a[0:3], v[230:233]
	v_mfma_f32_16x16x32_f16 v[234:237], a[24:27], a[0:3], v[234:237]
	v_mfma_f32_16x16x32_f16 v[238:241], a[32:35], a[0:3], v[238:241]
	v_mfma_f32_16x16x32_f16 v[242:245], a[40:43], a[0:3], v[242:245]
	v_mfma_f32_16x16x32_f16 v[246:249], a[48:51], a[0:3], v[246:249]
	v_mfma_f32_16x16x32_f16 v[250:253], a[56:59], a[0:3], v[250:253]
	v_mfma_f32_16x16x32_f16 a[0:3], a[72:75], a[0:3], a[8:11]
	v_mfma_f32_16x16x32_f16 a[208:211], a[20:23], a[92:95], a[100:103]
	v_mfma_f32_16x16x32_f16 a[232:235], a[68:71], a[92:95], a[96:99]
	v_mfma_f32_16x16x32_f16 a[240:243], a[20:23], a[84:87], a[176:179]
	v_mfma_f32_16x16x32_f16 a[252:255], a[44:47], a[84:87], a[88:91]
	v_mfma_f32_16x16x32_f16 v[182:185], a[52:55], a[84:87], v[182:185]
	v_mfma_f32_16x16x32_f16 v[186:189], a[60:63], a[84:87], v[186:189]
	v_mfma_f32_16x16x32_f16 v[190:193], a[68:71], a[84:87], v[190:193]
	v_mfma_f32_16x16x32_f16 v[194:197], a[76:79], a[84:87], v[194:197]
	v_mfma_f32_16x16x32_f16 v[198:201], a[20:23], a[12:15], v[198:201]
	v_mfma_f32_16x16x32_f16 v[202:205], a[28:31], a[12:15], v[202:205]
	v_mfma_f32_16x16x32_f16 v[206:209], a[36:39], a[12:15], v[206:209]
	v_mfma_f32_16x16x32_f16 v[210:213], a[44:47], a[12:15], v[210:213]
	v_mfma_f32_16x16x32_f16 v[214:217], a[52:55], a[12:15], v[214:217]
	v_mfma_f32_16x16x32_f16 v[218:221], a[60:63], a[12:15], v[218:221]
	v_mfma_f32_16x16x32_f16 v[222:225], a[68:71], a[12:15], v[222:225]
	v_mfma_f32_16x16x32_f16 v[226:229], a[76:79], a[12:15], v[226:229]
	v_mfma_f32_16x16x32_f16 v[230:233], a[20:23], a[4:7], v[230:233]
	v_mfma_f32_16x16x32_f16 v[234:237], a[28:31], a[4:7], v[234:237]
	v_mfma_f32_16x16x32_f16 v[238:241], a[36:39], a[4:7], v[238:241]
	v_mfma_f32_16x16x32_f16 v[242:245], a[44:47], a[4:7], v[242:245]
	v_mfma_f32_16x16x32_f16 v[246:249], a[52:55], a[4:7], v[246:249]
	v_mfma_f32_16x16x32_f16 v[250:253], a[60:63], a[4:7], v[250:253]
	v_mfma_f32_16x16x32_f16 a[156:159], a[76:79], a[4:7], a[0:3]
	s_nop 2
	ds_read_b128 a[0:3], v5 offset:33792
	ds_read_b128 a[4:7], v5 offset:33856
	ds_read_b128 a[8:11], v5 offset:34048
	ds_read_b128 a[12:15], v5 offset:34112
	ds_read_b128 a[16:19], v5 offset:34304
	ds_read_b128 a[20:23], v5 offset:34368
	ds_read_b128 a[24:27], v5 offset:34560
	ds_read_b128 a[28:31], v5 offset:34624
	ds_read_b128 a[32:35], v5 offset:50688
	ds_read_b128 a[36:39], v5 offset:50752
	ds_read_b128 a[40:43], v5 offset:50944
	ds_read_b128 a[44:47], v5 offset:51008
	ds_read_b128 a[48:51], v5 offset:51200
	ds_read_b128 a[52:55], v5 offset:51264
	ds_read_b128 a[56:59], v5 offset:51456
	ds_read_b128 a[60:63], v5 offset:51520
	ds_read_b128 a[64:67], v0
	ds_read_b128 a[68:71], v0 offset:64
	ds_read_b128 a[72:75], v0 offset:256
	ds_read_b128 a[76:79], v0 offset:320
	ds_read_b128 a[80:83], v0 offset:512
	ds_read_b128 a[84:87], v0 offset:576
	ds_read_b128 a[88:91], v0 offset:768
	ds_read_b128 a[92:95], v0 offset:832
	ds_read_b128 a[96:99], v0 offset:16896
	ds_read_b128 a[100:103], v0 offset:16960
	ds_read_b128 a[104:107], v0 offset:17152
	ds_read_b128 a[108:111], v0 offset:17216
	ds_read_b128 a[112:115], v0 offset:17408
	ds_read_b128 a[116:119], v0 offset:17472
	ds_read_b128 a[120:123], v0 offset:17664
	ds_read_b128 a[124:127], v0 offset:17728
	s_waitcnt vmcnt(0) lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[0:3], a[96:99], a[0:3], v[20:23]
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v18, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[22:25], a[100:103], a[4:7], v[0:3]
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v14, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[0:3], a[104:107], a[0:3], v[26:29]
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v254, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[26:29], a[108:111], a[4:7], v[0:3]
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v255, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[0:3], a[112:115], a[0:3], v[30:33]
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v7, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[30:33], a[116:119], a[4:7], v[0:3]
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v8, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[0:3], v[34:37]
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v9, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[34:37], a[124:127], a[4:7], v[0:3]
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v10, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[8:11], v[38:41]
	s_and_b32 s1, s21, 0xffff
	s_mov_b32 s0, s20
	s_mov_b32 m0, s47
	v_mfma_f32_16x16x32_f16 v[38:41], a[68:71], a[12:15], v[0:3]
	buffer_load_dwordx4 v11, s[0:3], 0 offen lds
	s_mov_b32 m0, s48
	s_add_u32 s20, s20, 0x100
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[8:11], v[42:45]
	buffer_load_dwordx4 v12, s[0:3], 0 offen lds
	s_mov_b32 m0, s49
	s_addc_u32 s21, s21, 0
	v_mfma_f32_16x16x32_f16 v[42:45], a[76:79], a[12:15], v[0:3]
	buffer_load_dwordx4 v13, s[0:3], 0 offen lds
	s_mov_b32 m0, s50
	s_add_u32 s22, s22, 0x100
	v_mfma_f32_16x16x32_f16 v[0:3], a[80:83], a[8:11], v[46:49]
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_mov_b32 m0, s51
	s_addc_u32 s23, s23, 0
	v_mfma_f32_16x16x32_f16 v[46:49], a[84:87], a[12:15], v[0:3]
	buffer_load_dwordx4 v15, s[0:3], 0 offen lds
	s_mov_b32 m0, s52
	s_add_i32 s55, s55, 2
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[8:11], v[50:53]
	buffer_load_dwordx4 v16, s[0:3], 0 offen lds
	s_mov_b32 m0, s53
	s_cmpk_lt_u32 s55, 0x7c
	v_mfma_f32_16x16x32_f16 v[50:53], a[92:95], a[12:15], v[0:3]
	buffer_load_dwordx4 v17, s[0:3], 0 offen lds
	s_mov_b32 m0, s54
	v_mfma_f32_16x16x32_f16 v[0:3], a[96:99], a[8:11], v[54:57]
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[54:57], a[100:103], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[104:107], a[8:11], v[58:61]
	v_mfma_f32_16x16x32_f16 v[58:61], a[108:111], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[112:115], a[8:11], v[62:65]
	v_mfma_f32_16x16x32_f16 v[62:65], a[116:119], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[8:11], v[66:69]
	v_mfma_f32_16x16x32_f16 v[66:69], a[124:127], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[16:19], v[70:73]
	v_mfma_f32_16x16x32_f16 v[70:73], a[68:71], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[16:19], v[74:77]
	v_mfma_f32_16x16x32_f16 v[74:77], a[76:79], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[80:83], a[16:19], v[78:81]
	v_mfma_f32_16x16x32_f16 v[78:81], a[84:87], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[16:19], v[82:85]
	v_mfma_f32_16x16x32_f16 v[82:85], a[92:95], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[96:99], a[16:19], v[86:89]
	v_mfma_f32_16x16x32_f16 v[86:89], a[100:103], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[104:107], a[16:19], v[90:93]
	v_mfma_f32_16x16x32_f16 v[90:93], a[108:111], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[112:115], a[16:19], v[94:97]
	v_mfma_f32_16x16x32_f16 v[94:97], a[116:119], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[16:19], v[98:101]
	v_mfma_f32_16x16x32_f16 v[98:101], a[124:127], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[24:27], v[102:105]
	v_mfma_f32_16x16x32_f16 v[102:105], a[68:71], a[28:31], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[96:99], a[40:43], v[182:185]
	v_mfma_f32_16x16x32_f16 v[182:185], a[100:103], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[104:107], a[40:43], v[186:189]
	v_mfma_f32_16x16x32_f16 v[186:189], a[108:111], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[112:115], a[40:43], v[190:193]
	v_mfma_f32_16x16x32_f16 v[190:193], a[116:119], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[40:43], v[194:197]
	v_mfma_f32_16x16x32_f16 v[194:197], a[124:127], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[48:51], v[198:201]
	v_mfma_f32_16x16x32_f16 v[198:201], a[68:71], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[48:51], v[202:205]
	v_mfma_f32_16x16x32_f16 v[202:205], a[76:79], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[80:83], a[48:51], v[206:209]
	v_mfma_f32_16x16x32_f16 v[206:209], a[84:87], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[48:51], v[210:213]
	v_mfma_f32_16x16x32_f16 v[210:213], a[92:95], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[96:99], a[48:51], v[214:217]
	v_mfma_f32_16x16x32_f16 v[214:217], a[100:103], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[104:107], a[48:51], v[218:221]
	v_mfma_f32_16x16x32_f16 v[218:221], a[108:111], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[112:115], a[48:51], v[222:225]
	v_mfma_f32_16x16x32_f16 v[222:225], a[116:119], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[120:123], a[48:51], v[226:229]
	v_mfma_f32_16x16x32_f16 v[226:229], a[124:127], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], a[56:59], v[230:233]
	v_mfma_f32_16x16x32_f16 a[164:167], a[72:75], a[0:3], a[164:167]
	v_mfma_f32_16x16x32_f16 v[230:233], a[68:71], a[60:63], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], a[56:59], v[234:237]
	v_mfma_f32_16x16x32_f16 a[176:179], a[76:79], a[4:7], a[164:167]
	s_nop 4
	v_accvgpr_mov_b32 a164, a168
	v_accvgpr_mov_b32 a165, a169
	v_accvgpr_mov_b32 a166, a170
	v_accvgpr_mov_b32 a167, a171
	v_accvgpr_mov_b32 a168, a172
	v_accvgpr_mov_b32 a169, a173
	v_accvgpr_mov_b32 a170, a174
	v_accvgpr_mov_b32 a171, a175
	v_mfma_f32_16x16x32_f16 a[160:163], a[64:67], a[0:3], a[160:163]
	v_accvgpr_read_b32 v8, a176
	v_accvgpr_read_b32 v9, a177
	v_accvgpr_read_b32 v10, a178
	v_mfma_f32_16x16x32_f16 a[164:167], a[80:83], a[0:3], a[164:167]
	v_accvgpr_read_b32 v11, a179
	v_mfma_f32_16x16x32_f16 a[168:171], a[88:91], a[0:3], a[168:171]
	v_mfma_f32_16x16x32_f16 v[234:237], a[76:79], a[60:63], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[80:83], a[56:59], v[238:241]
	v_mfma_f32_16x16x32_f16 a[160:163], a[68:71], a[4:7], a[160:163]
	v_mfma_f32_16x16x32_f16 a[164:167], a[84:87], a[4:7], a[164:167]
	v_mfma_f32_16x16x32_f16 a[168:171], a[92:95], a[4:7], a[168:171]
	v_mfma_f32_16x16x32_f16 a[0:3], a[72:75], a[24:27], a[180:183]
	s_nop 5
	v_accvgpr_read_b32 v14, a164
	v_accvgpr_read_b32 v15, a165
	v_accvgpr_read_b32 v16, a166
	v_mfma_f32_16x16x32_f16 a[4:7], a[80:83], a[24:27], a[184:187]
	v_accvgpr_read_b32 v18, a168
	v_accvgpr_read_b32 v19, a169
	v_accvgpr_read_b32 v20, a170
	v_mfma_f32_16x16x32_f16 a[8:11], a[88:91], a[24:27], a[188:191]
	v_accvgpr_read_b32 v21, a171
	v_accvgpr_read_b32 v17, a167
	v_mfma_f32_16x16x32_f16 a[12:15], a[96:99], a[24:27], a[192:195]
	v_mfma_f32_16x16x32_f16 a[16:19], a[104:107], a[24:27], a[196:199]
	v_mfma_f32_16x16x32_f16 a[20:23], a[112:115], a[24:27], a[200:203]
	v_mfma_f32_16x16x32_f16 a[24:27], a[120:123], a[24:27], a[204:207]
	v_mfma_f32_16x16x32_f16 v[238:241], a[84:87], a[60:63], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[88:91], a[56:59], v[242:245]
	v_mfma_f32_16x16x32_f16 a[0:3], a[76:79], a[28:31], a[0:3]
	v_mfma_f32_16x16x32_f16 a[4:7], a[84:87], a[28:31], a[4:7]
	v_mfma_f32_16x16x32_f16 a[8:11], a[92:95], a[28:31], a[8:11]
	s_nop 5
	v_accvgpr_mov_b32 a179, a3
	v_accvgpr_mov_b32 a178, a2
	v_accvgpr_mov_b32 a177, a1
	v_mfma_f32_16x16x32_f16 a[12:15], a[100:103], a[28:31], a[12:15]
	v_accvgpr_mov_b32 a176, a0
	v_mfma_f32_16x16x32_f16 a[16:19], a[108:111], a[28:31], a[16:19]
	v_mfma_f32_16x16x32_f16 a[20:23], a[116:119], a[28:31], a[20:23]
	v_mfma_f32_16x16x32_f16 a[24:27], a[124:127], a[28:31], a[24:27]
	v_mfma_f32_16x16x32_f16 a[28:31], a[64:67], a[32:35], a[208:211]
	v_mfma_f32_16x16x32_f16 v[242:245], a[92:95], a[60:63], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[96:99], a[56:59], v[246:249]
	v_mfma_f32_16x16x32_f16 a[172:175], a[72:75], a[32:35], a[212:215]
	v_mfma_f32_16x16x32_f16 a[180:183], a[80:83], a[32:35], a[216:219]
	v_mfma_f32_16x16x32_f16 a[184:187], a[88:91], a[32:35], a[220:223]
	v_mfma_f32_16x16x32_f16 a[188:191], a[96:99], a[32:35], a[224:227]
	v_mfma_f32_16x16x32_f16 a[192:195], a[104:107], a[32:35], a[228:231]
	v_mfma_f32_16x16x32_f16 a[196:199], a[112:115], a[32:35], a[232:235]
	v_mfma_f32_16x16x32_f16 a[32:35], a[120:123], a[32:35], a[236:239]
	v_mfma_f32_16x16x32_f16 a[28:31], a[68:71], a[36:39], a[28:31]
	v_mfma_f32_16x16x32_f16 a[200:203], a[72:75], a[40:43], a[244:247]
	v_mfma_f32_16x16x32_f16 a[204:207], a[80:83], a[40:43], a[248:251]
	v_mfma_f32_16x16x32_f16 a[208:211], a[88:91], a[40:43], a[252:255]
	v_mfma_f32_16x16x32_f16 v[246:249], a[100:103], a[60:63], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[104:107], a[56:59], v[250:253]
	v_mfma_f32_16x16x32_f16 a[212:215], a[76:79], a[36:39], a[172:175]
	v_mfma_f32_16x16x32_f16 a[216:219], a[84:87], a[36:39], a[180:183]
	v_mfma_f32_16x16x32_f16 a[220:223], a[92:95], a[36:39], a[184:187]
	v_mfma_f32_16x16x32_f16 a[224:227], a[100:103], a[36:39], a[188:191]
	s_nop 1
	v_accvgpr_mov_b32 a187, a7
	v_accvgpr_mov_b32 a186, a6
	v_accvgpr_mov_b32 a185, a5
	v_mfma_f32_16x16x32_f16 a[228:231], a[108:111], a[36:39], a[192:195]
	v_accvgpr_mov_b32 a191, a11
	v_accvgpr_mov_b32 a190, a10
	v_accvgpr_mov_b32 a189, a9
	v_mfma_f32_16x16x32_f16 a[232:235], a[116:119], a[36:39], a[196:199]
	v_accvgpr_mov_b32 a195, a15
	v_accvgpr_mov_b32 a194, a14
	v_accvgpr_mov_b32 a193, a13
	v_mfma_f32_16x16x32_f16 a[236:239], a[124:127], a[36:39], a[32:35]
	v_accvgpr_mov_b32 a199, a19
	v_accvgpr_mov_b32 a198, a18
	v_accvgpr_mov_b32 a197, a17
	v_mfma_f32_16x16x32_f16 a[36:39], a[64:67], a[40:43], a[240:243]
	v_accvgpr_mov_b32 a196, a16
	v_accvgpr_mov_b32 a192, a12
	v_accvgpr_mov_b32 a188, a8
	v_mfma_f32_16x16x32_f16 a[152:155], a[112:115], a[56:59], a[152:155]
	v_accvgpr_mov_b32 a184, a4
	v_mfma_f32_16x16x32_f16 a[156:159], a[120:123], a[56:59], a[156:159]
	v_mfma_f32_16x16x32_f16 a[244:247], a[76:79], a[44:47], a[200:203]
	v_mfma_f32_16x16x32_f16 a[248:251], a[84:87], a[44:47], a[204:207]
	s_nop 1
	v_accvgpr_mov_b32 a203, a23
	v_accvgpr_mov_b32 a202, a22
	v_accvgpr_mov_b32 a201, a21
	v_mfma_f32_16x16x32_f16 a[252:255], a[92:95], a[44:47], a[208:211]
	v_accvgpr_mov_b32 a207, a27
	v_accvgpr_mov_b32 a206, a26
	v_accvgpr_mov_b32 a205, a25
	v_mfma_f32_16x16x32_f16 v[250:253], a[108:111], a[60:63], v[0:3]
	v_accvgpr_mov_b32 a211, a31
	v_accvgpr_mov_b32 a210, a30
	v_accvgpr_mov_b32 a209, a29
	v_accvgpr_read_b32 v0, a150
	v_mfma_f32_16x16x32_f16 a[240:243], a[68:71], a[44:47], a[36:39]
	v_accvgpr_mov_b32 a208, a28
	v_accvgpr_mov_b32 a204, a24
	v_accvgpr_mov_b32 a200, a20
	v_mfma_f32_16x16x32_f16 a[152:155], a[116:119], a[60:63], a[152:155]
	v_mfma_f32_16x16x32_f16 a[156:159], a[124:127], a[60:63], a[156:159]
	ds_read_b128 a[120:123], v5
	ds_read_b128 a[124:127], v5 offset:64
	ds_read_b128 a[112:115], v5 offset:256
	ds_read_b128 a[116:119], v5 offset:320
	ds_read_b128 a[104:107], v5 offset:512
	ds_read_b128 a[108:111], v5 offset:576
	ds_read_b128 a[96:99], v5 offset:768
	ds_read_b128 a[100:103], v5 offset:832
	ds_read_b128 a[88:91], v5 offset:16896
	ds_read_b128 a[92:95], v5 offset:16960
	ds_read_b128 a[80:83], v5 offset:17152
	ds_read_b128 a[84:87], v5 offset:17216
	ds_read_b128 a[8:11], v5 offset:17408
	ds_read_b128 a[12:15], v5 offset:17472
	ds_read_b128 a[0:3], v5 offset:17664
	ds_read_b128 a[4:7], v5 offset:17728
	ds_read_b128 a[16:19], v0
	ds_read_b128 a[20:23], v0 offset:64
	ds_read_b128 a[24:27], v0 offset:256
	ds_read_b128 a[28:31], v0 offset:320
	ds_read_b128 a[32:35], v0 offset:512
	ds_read_b128 a[36:39], v0 offset:576
	ds_read_b128 a[40:43], v0 offset:768
	ds_read_b128 a[44:47], v0 offset:832
	ds_read_b128 a[48:51], v0 offset:16896
	ds_read_b128 a[52:55], v0 offset:16960
	ds_read_b128 a[56:59], v0 offset:17152
	ds_read_b128 a[60:63], v0 offset:17216
	ds_read_b128 a[64:67], v0 offset:17408
	ds_read_b128 a[68:71], v0 offset:17472
	ds_read_b128 a[72:75], v0 offset:17664
	ds_read_b128 a[76:79], v0 offset:17728
	v_accvgpr_read_b32 v2, a160
	v_accvgpr_read_b32 v3, a161
	v_accvgpr_read_b32 v4, a162
	v_accvgpr_read_b32 v5, a163
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[0:3], a[16:19], a[120:123], v[2:5]
	s_lshr_b32 s0, s4, 2
	s_and_b32 s1, s24, 0x80
	s_mov_b32 s55, 0x27000
	v_mfma_f32_16x16x32_f16 v[4:7], a[20:23], a[124:127], v[0:3]
	s_mov_b32 s54, 0x7ffffffe
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[0:3], a[24:27], a[120:123], v[8:11]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[8:11], a[32:35], a[120:123], v[14:17]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_f16 v[14:17], a[40:43], a[120:123], v[18:21]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[18:21], a[48:51], a[120:123], v[22:25]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[22:25], a[56:59], a[120:123], v[26:29]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[26:29], a[64:67], a[120:123], v[30:33]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[30:33], a[72:75], a[120:123], v[34:37]
	v_mfma_f32_16x16x32_f16 v[34:37], a[16:19], a[112:115], v[38:41]
	v_mfma_f32_16x16x32_f16 v[38:41], a[24:27], a[112:115], v[42:45]
	v_mfma_f32_16x16x32_f16 v[42:45], a[32:35], a[112:115], v[46:49]
	v_mfma_f32_16x16x32_f16 v[46:49], a[40:43], a[112:115], v[50:53]
	v_mfma_f32_16x16x32_f16 v[50:53], a[48:51], a[112:115], v[54:57]
	v_mfma_f32_16x16x32_f16 v[54:57], a[56:59], a[112:115], v[58:61]
	v_mfma_f32_16x16x32_f16 v[58:61], a[64:67], a[112:115], v[62:65]
	v_mfma_f32_16x16x32_f16 v[62:65], a[72:75], a[112:115], v[66:69]
	v_mfma_f32_16x16x32_f16 v[66:69], a[16:19], a[104:107], v[70:73]
	v_mfma_f32_16x16x32_f16 v[70:73], a[24:27], a[104:107], v[74:77]
	v_mfma_f32_16x16x32_f16 v[74:77], a[32:35], a[104:107], v[78:81]
	v_mfma_f32_16x16x32_f16 v[78:81], a[40:43], a[104:107], v[82:85]
	v_mfma_f32_16x16x32_f16 v[82:85], a[48:51], a[104:107], v[86:89]
	v_mfma_f32_16x16x32_f16 v[86:89], a[56:59], a[104:107], v[90:93]
	v_mfma_f32_16x16x32_f16 v[90:93], a[64:67], a[104:107], v[94:97]
	v_mfma_f32_16x16x32_f16 v[94:97], a[72:75], a[104:107], v[98:101]
	v_mfma_f32_16x16x32_f16 v[98:101], a[16:19], a[96:99], v[102:105]
	v_mfma_f32_16x16x32_f16 v[102:105], a[48:51], a[80:83], v[182:185]
	v_mfma_f32_16x16x32_f16 v[102:105], a[52:55], a[84:87], v[102:105]
	v_mfma_f32_16x16x32_f16 v[34:37], a[20:23], a[116:119], v[34:37]
	v_mfma_f32_16x16x32_f16 v[38:41], a[28:31], a[116:119], v[38:41]
	v_mfma_f32_16x16x32_f16 v[42:45], a[36:39], a[116:119], v[42:45]
	v_mfma_f32_16x16x32_f16 v[46:49], a[44:47], a[116:119], v[46:49]
	v_mfma_f32_16x16x32_f16 v[50:53], a[52:55], a[116:119], v[50:53]
	v_mfma_f32_16x16x32_f16 v[54:57], a[60:63], a[116:119], v[54:57]
	v_mfma_f32_16x16x32_f16 v[58:61], a[68:71], a[116:119], v[58:61]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[62:65], a[76:79], a[116:119], v[62:65]
	v_mfma_f32_16x16x32_f16 a[116:119], a[48:51], a[96:99], a[192:195]
	s_nop 2
	v_accvgpr_write_b32 a195, v105
	v_accvgpr_write_b32 a194, v104
	v_accvgpr_write_b32 a193, v103
	v_accvgpr_write_b32 a192, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[56:59], a[80:83], v[186:189]
	v_mfma_f32_16x16x32_f16 v[102:105], a[60:63], a[84:87], v[102:105]
	v_mfma_f32_16x16x32_f16 a[120:123], a[56:59], a[96:99], a[196:199]
	v_mfma_f32_16x16x32_f16 v[0:3], a[28:31], a[124:127], v[0:3]
	s_nop 5
	v_accvgpr_write_b32 a199, v105
	v_accvgpr_write_b32 a198, v104
	v_accvgpr_write_b32 a197, v103
	v_accvgpr_write_b32 a196, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[64:67], a[80:83], v[190:193]
	v_mfma_f32_16x16x32_f16 v[102:105], a[68:71], a[84:87], v[102:105]
	v_mfma_f32_16x16x32_f16 v[10:13], a[36:39], a[124:127], v[8:11]
	v_mfma_f32_16x16x32_f16 v[14:17], a[44:47], a[124:127], v[14:17]
	s_nop 1
	v_accvgpr_read_b32 v8, a135
	v_mfma_f32_16x16x32_f16 v[18:21], a[52:55], a[124:127], v[18:21]
	v_mfma_f32_16x16x32_f16 v[22:25], a[60:63], a[124:127], v[22:25]
	v_mfma_f32_16x16x32_f16 v[26:29], a[68:71], a[124:127], v[26:29]
	v_mfma_f32_16x16x32_f16 v[30:33], a[76:79], a[124:127], v[30:33]
	v_mfma_f32_16x16x32_f16 a[124:127], a[64:67], a[96:99], a[200:203]
	s_nop 2
	v_accvgpr_write_b32 a203, v105
	v_accvgpr_write_b32 a202, v104
	v_accvgpr_write_b32 a201, v103
	v_accvgpr_write_b32 a200, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[72:75], a[80:83], v[194:197]
	v_mfma_f32_16x16x32_f16 v[102:105], a[76:79], a[84:87], v[102:105]
	v_mfma_f32_16x16x32_f16 v[66:69], a[20:23], a[108:111], v[66:69]
	v_mfma_f32_16x16x32_f16 v[70:73], a[28:31], a[108:111], v[70:73]
	v_mfma_f32_16x16x32_f16 v[74:77], a[36:39], a[108:111], v[74:77]
	v_mfma_f32_16x16x32_f16 v[78:81], a[44:47], a[108:111], v[78:81]
	v_mfma_f32_16x16x32_f16 v[82:85], a[52:55], a[108:111], v[82:85]
	v_mfma_f32_16x16x32_f16 v[86:89], a[60:63], a[108:111], v[86:89]
	v_mfma_f32_16x16x32_f16 v[90:93], a[68:71], a[108:111], v[90:93]
	v_mfma_f32_16x16x32_f16 v[94:97], a[76:79], a[108:111], v[94:97]
	v_mfma_f32_16x16x32_f16 a[104:107], a[24:27], a[96:99], a[176:179]
	v_mfma_f32_16x16x32_f16 a[108:111], a[32:35], a[96:99], a[184:187]
	v_mfma_f32_16x16x32_f16 a[112:115], a[40:43], a[96:99], a[188:191]
	v_mfma_f32_16x16x32_f16 a[96:99], a[72:75], a[96:99], a[204:207]
	s_nop 2
	v_accvgpr_write_b32 a207, v105
	v_accvgpr_write_b32 a206, v104
	v_accvgpr_write_b32 a205, v103
	v_accvgpr_write_b32 a204, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[16:19], a[8:11], v[198:201]
	v_mfma_f32_16x16x32_f16 v[102:105], a[20:23], a[12:15], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[20:23], a[100:103], v[98:101]
	v_mfma_f32_16x16x32_f16 a[104:107], a[28:31], a[100:103], a[104:107]
	v_mfma_f32_16x16x32_f16 a[108:111], a[36:39], a[100:103], a[108:111]
	v_mfma_f32_16x16x32_f16 a[112:115], a[44:47], a[100:103], a[112:115]
	v_mfma_f32_16x16x32_f16 a[116:119], a[52:55], a[100:103], a[116:119]
	v_mfma_f32_16x16x32_f16 a[120:123], a[60:63], a[100:103], a[120:123]
	v_mfma_f32_16x16x32_f16 a[124:127], a[68:71], a[100:103], a[124:127]
	v_mfma_f32_16x16x32_f16 a[96:99], a[76:79], a[100:103], a[96:99]
	v_mfma_f32_16x16x32_f16 a[100:103], a[16:19], a[88:91], a[208:211]
	s_nop 2
	v_accvgpr_write_b32 a211, v105
	v_accvgpr_write_b32 a210, v104
	v_accvgpr_write_b32 a209, v103
	v_accvgpr_write_b32 a208, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[24:27], a[8:11], v[202:205]
	v_mfma_f32_16x16x32_f16 v[102:105], a[28:31], a[12:15], v[102:105]
	v_mfma_f32_16x16x32_f16 a[136:139], a[24:27], a[88:91], a[212:215]
	v_mfma_f32_16x16x32_f16 a[140:143], a[32:35], a[88:91], a[216:219]
	s_nop 5
	v_accvgpr_write_b32 a215, v105
	v_accvgpr_write_b32 a214, v104
	v_accvgpr_write_b32 a213, v103
	v_accvgpr_write_b32 a212, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[32:35], a[8:11], v[206:209]
	v_mfma_f32_16x16x32_f16 v[102:105], a[36:39], a[12:15], v[102:105]
	v_mfma_f32_16x16x32_f16 a[144:147], a[40:43], a[88:91], a[220:223]
	v_mfma_f32_16x16x32_f16 a[160:163], a[48:51], a[88:91], a[224:227]
	s_nop 5
	v_accvgpr_write_b32 a219, v105
	v_accvgpr_write_b32 a218, v104
	v_accvgpr_write_b32 a217, v103
	v_accvgpr_write_b32 a216, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[40:43], a[8:11], v[210:213]
	v_mfma_f32_16x16x32_f16 v[102:105], a[44:47], a[12:15], v[102:105]
	v_mfma_f32_16x16x32_f16 a[164:167], a[56:59], a[88:91], a[228:231]
	v_mfma_f32_16x16x32_f16 a[168:171], a[64:67], a[88:91], a[232:235]
	s_nop 5
	v_accvgpr_write_b32 a223, v105
	v_accvgpr_write_b32 a222, v104
	v_accvgpr_write_b32 a221, v103
	v_accvgpr_write_b32 a220, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[48:51], a[8:11], v[214:217]
	v_mfma_f32_16x16x32_f16 v[102:105], a[52:55], a[12:15], v[102:105]
	v_mfma_f32_16x16x32_f16 a[172:175], a[72:75], a[88:91], a[236:239]
	v_mfma_f32_16x16x32_f16 a[176:179], a[16:19], a[80:83], a[240:243]
	s_nop 5
	v_accvgpr_write_b32 a227, v105
	v_accvgpr_write_b32 a226, v104
	v_accvgpr_write_b32 a225, v103
	v_accvgpr_write_b32 a224, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[56:59], a[8:11], v[218:221]
	v_mfma_f32_16x16x32_f16 v[102:105], a[60:63], a[12:15], v[102:105]
	v_mfma_f32_16x16x32_f16 a[180:183], a[24:27], a[80:83], a[244:247]
	v_mfma_f32_16x16x32_f16 a[184:187], a[32:35], a[80:83], a[248:251]
	s_nop 5
	v_accvgpr_write_b32 a231, v105
	v_accvgpr_write_b32 a230, v104
	v_accvgpr_write_b32 a229, v103
	v_accvgpr_write_b32 a228, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[64:67], a[8:11], v[222:225]
	v_mfma_f32_16x16x32_f16 v[102:105], a[68:71], a[12:15], v[102:105]
	v_mfma_f32_16x16x32_f16 a[188:191], a[40:43], a[80:83], a[252:255]
	v_mfma_f32_16x16x32_f16 a[100:103], a[20:23], a[92:95], a[100:103]
	s_nop 5
	v_accvgpr_write_b32 a235, v105
	v_accvgpr_write_b32 a234, v104
	v_accvgpr_write_b32 a233, v103
	v_accvgpr_write_b32 a232, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[72:75], a[8:11], v[226:229]
	v_accvgpr_mov_b32 a8, a152
	v_accvgpr_mov_b32 a9, a153
	v_accvgpr_mov_b32 a10, a154
	v_mfma_f32_16x16x32_f16 v[102:105], a[76:79], a[12:15], v[102:105]
	v_accvgpr_mov_b32 a11, a155
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[8:11], a[64:67], a[0:3], a[8:11]
	v_mfma_f32_16x16x32_f16 a[152:155], a[68:71], a[4:7], a[8:11]
	s_nop 2
	v_accvgpr_write_b32 a239, v105
	v_accvgpr_write_b32 a238, v104
	v_accvgpr_write_b32 a237, v103
	v_accvgpr_write_b32 a236, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[16:19], a[0:3], v[230:233]
	v_accvgpr_mov_b32 a8, a156
	v_accvgpr_mov_b32 a9, a157
	v_accvgpr_mov_b32 a10, a158
	v_mfma_f32_16x16x32_f16 v[102:105], a[20:23], a[4:7], v[102:105]
	v_accvgpr_mov_b32 a11, a159
	v_mfma_f32_16x16x32_f16 a[136:139], a[28:31], a[92:95], a[136:139]
	v_mfma_f32_16x16x32_f16 a[140:143], a[36:39], a[92:95], a[140:143]
	s_nop 4
	v_accvgpr_write_b32 a243, v105
	v_accvgpr_write_b32 a242, v104
	v_accvgpr_write_b32 a241, v103
	v_accvgpr_write_b32 a240, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[24:27], a[0:3], v[234:237]
	v_mfma_f32_16x16x32_f16 v[102:105], a[28:31], a[4:7], v[102:105]
	v_mfma_f32_16x16x32_f16 a[144:147], a[44:47], a[92:95], a[144:147]
	v_mfma_f32_16x16x32_f16 a[160:163], a[52:55], a[92:95], a[160:163]
	s_nop 5
	v_accvgpr_write_b32 a247, v105
	v_accvgpr_write_b32 a246, v104
	v_accvgpr_write_b32 a245, v103
	v_accvgpr_write_b32 a244, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[32:35], a[0:3], v[238:241]
	v_mfma_f32_16x16x32_f16 v[102:105], a[36:39], a[4:7], v[102:105]
	v_mfma_f32_16x16x32_f16 a[164:167], a[60:63], a[92:95], a[164:167]
	v_mfma_f32_16x16x32_f16 a[168:171], a[68:71], a[92:95], a[168:171]
	s_nop 5
	v_accvgpr_write_b32 a251, v105
	v_accvgpr_write_b32 a250, v104
	v_accvgpr_write_b32 a249, v103
	v_accvgpr_write_b32 a248, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[40:43], a[0:3], v[242:245]
	v_mfma_f32_16x16x32_f16 v[102:105], a[44:47], a[4:7], v[102:105]
	v_mfma_f32_16x16x32_f16 a[172:175], a[76:79], a[92:95], a[172:175]
	v_mfma_f32_16x16x32_f16 a[176:179], a[20:23], a[84:87], a[176:179]
	s_nop 5
	v_accvgpr_write_b32 a255, v105
	v_accvgpr_write_b32 a254, v104
	v_accvgpr_write_b32 a253, v103
	v_accvgpr_write_b32 a252, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[48:51], a[0:3], v[246:249]
	v_mfma_f32_16x16x32_f16 v[102:105], a[52:55], a[4:7], v[102:105]
	v_mfma_f32_16x16x32_f16 a[180:183], a[28:31], a[84:87], a[180:183]
	v_mfma_f32_16x16x32_f16 a[184:187], a[36:39], a[84:87], a[184:187]
	s_nop 5
	v_accvgpr_write_b32 a131, v105
	v_accvgpr_write_b32 a130, v104
	v_accvgpr_write_b32 a129, v103
	v_accvgpr_write_b32 a128, v102
	v_mfma_f32_16x16x32_f16 v[102:105], a[56:59], a[0:3], v[250:253]
	v_mfma_f32_16x16x32_f16 v[102:105], a[60:63], a[4:7], v[102:105]
	v_mfma_f32_16x16x32_f16 a[0:3], a[72:75], a[0:3], a[8:11]
	v_mfma_f32_16x16x32_f16 a[188:191], a[44:47], a[84:87], a[188:191]
	s_nop 5
	v_accvgpr_write_b32 a151, v105
	v_accvgpr_write_b32 a150, v104
	v_accvgpr_write_b32 a149, v103
	v_accvgpr_write_b32 a148, v102
	v_mfma_f32_16x16x32_f16 a[156:159], a[76:79], a[4:7], a[0:3]
	ds_read_b128 v[102:105], v8 offset:33792
	ds_read_b128 v[106:109], v8 offset:33856
	ds_read_b128 v[110:113], v8 offset:34048
	ds_read_b128 v[114:117], v8 offset:34112
	ds_read_b128 v[142:145], v8 offset:34304
	ds_read_b128 v[146:149], v8 offset:34368
	ds_read_b128 v[174:177], v8 offset:34560
	ds_read_b128 v[178:181], v8 offset:34624
	ds_read_b128 a[88:91], v8 offset:50688
	ds_read_b128 a[92:95], v8 offset:50752
	ds_read_b128 a[80:83], v8 offset:50944
	ds_read_b128 a[84:87], v8 offset:51008
	ds_read_b128 a[8:11], v8 offset:51200
	ds_read_b128 a[12:15], v8 offset:51264
	ds_read_b128 a[0:3], v8 offset:51456
	ds_read_b128 a[4:7], v8 offset:51520
	v_mov_b32_e32 v8, v118
	v_add_u32_e32 v8, 0x18be0, v8
	ds_read_b128 a[16:19], v8
	ds_read_b128 a[20:23], v8 offset:64
	ds_read_b128 a[24:27], v8 offset:256
	ds_read_b128 a[28:31], v8 offset:320
	ds_read_b128 a[32:35], v8 offset:512
	ds_read_b128 a[36:39], v8 offset:576
	ds_read_b128 a[40:43], v8 offset:768
	ds_read_b128 a[44:47], v8 offset:832
	ds_read_b128 a[48:51], v8 offset:16896
	ds_read_b128 a[52:55], v8 offset:16960
	ds_read_b128 a[56:59], v8 offset:17152
	ds_read_b128 a[60:63], v8 offset:17216
	ds_read_b128 a[64:67], v8 offset:17408
	ds_read_b128 a[68:71], v8 offset:17472
	ds_read_b128 a[72:75], v8 offset:17664
	ds_read_b128 a[76:79], v8 offset:17728
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[0:3], a[24:27], v[102:105], v[0:3]
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_f16 v[210:213], a[28:31], v[106:109], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[32:35], v[102:105], v[10:13]
	v_mfma_f32_16x16x32_f16 v[214:217], a[36:39], v[106:109], v[0:3]
	s_nop 5
	v_cvt_pk_f16_f32 v210, v210, v211
	v_cvt_pk_f16_f32 v211, v212, v213
	v_mfma_f32_16x16x32_f16 v[0:3], a[40:43], v[102:105], v[14:17]
	v_mfma_f32_16x16x32_f16 v[218:221], a[44:47], v[106:109], v[0:3]
	v_cvt_pk_f16_f32 v214, v214, v215
	v_cvt_pk_f16_f32 v215, v216, v217
	v_mfma_f32_16x16x32_f16 v[0:3], a[48:51], v[102:105], v[18:21]
	v_mfma_f32_16x16x32_f16 v[222:225], a[52:55], v[106:109], v[0:3]
	s_nop 3
	v_cvt_pk_f16_f32 v218, v218, v219
	v_cvt_pk_f16_f32 v219, v220, v221
	v_mfma_f32_16x16x32_f16 v[0:3], a[56:59], v[102:105], v[22:25]
	v_mfma_f32_16x16x32_f16 v[226:229], a[60:63], v[106:109], v[0:3]
	v_cvt_pk_f16_f32 v222, v222, v223
	v_cvt_pk_f16_f32 v223, v224, v225
	v_mfma_f32_16x16x32_f16 v[0:3], a[64:67], v[102:105], v[26:29]
	v_mfma_f32_16x16x32_f16 v[230:233], a[68:71], v[106:109], v[0:3]
	s_nop 3
	v_cvt_pk_f16_f32 v226, v226, v227
	v_cvt_pk_f16_f32 v227, v228, v229
	v_mfma_f32_16x16x32_f16 v[0:3], a[72:75], v[102:105], v[30:33]
	v_mfma_f32_16x16x32_f16 v[4:7], a[16:19], v[102:105], v[4:7]
	v_cvt_pk_f16_f32 v230, v230, v231
	v_cvt_pk_f16_f32 v231, v232, v233
	v_mfma_f32_16x16x32_f16 v[234:237], a[76:79], v[106:109], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[16:19], v[110:113], v[34:37]
	v_mfma_f32_16x16x32_f16 v[206:209], a[20:23], v[106:109], v[4:7]
	s_nop 5
	v_cvt_pk_f16_f32 v234, v234, v235
	v_cvt_pk_f16_f32 v235, v236, v237
	v_mfma_f32_16x16x32_f16 v[238:241], a[20:23], v[114:117], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[24:27], v[110:113], v[38:41]
	v_cvt_pk_f16_f32 v206, v206, v207
	v_cvt_pk_f16_f32 v207, v208, v209
	v_mfma_f32_16x16x32_f16 v[4:7], a[56:59], v[110:113], v[54:57]
	s_nop 3
	v_cvt_pk_f16_f32 v208, v238, v239
	v_cvt_pk_f16_f32 v209, v240, v241
	v_mfma_f32_16x16x32_f16 v[242:245], a[28:31], v[114:117], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[32:35], v[110:113], v[42:45]
	v_mfma_f32_16x16x32_f16 v[102:105], a[60:63], v[114:117], v[4:7]
	s_nop 5
	v_cvt_pk_f16_f32 v212, v242, v243
	v_cvt_pk_f16_f32 v213, v244, v245
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], v[110:113], v[58:61]
	v_mfma_f32_16x16x32_f16 v[246:249], a[36:39], v[114:117], v[0:3]
	v_cvt_pk_f16_f32 v228, v102, v103
	v_cvt_pk_f16_f32 v229, v104, v105
	v_mfma_f32_16x16x32_f16 v[0:3], a[40:43], v[110:113], v[46:49]
	v_mfma_f32_16x16x32_f16 v[106:109], a[68:71], v[114:117], v[4:7]
	s_nop 3
	v_cvt_pk_f16_f32 v216, v246, v247
	v_cvt_pk_f16_f32 v217, v248, v249
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], v[110:113], v[62:65]
	v_mfma_f32_16x16x32_f16 v[250:253], a[44:47], v[114:117], v[0:3]
	v_cvt_pk_f16_f32 v232, v106, v107
	v_cvt_pk_f16_f32 v233, v108, v109
	v_mfma_f32_16x16x32_f16 v[0:3], a[48:51], v[110:113], v[50:53]
	v_mfma_f32_16x16x32_f16 v[110:113], a[76:79], v[114:117], v[4:7]
	s_nop 3
	v_cvt_pk_f16_f32 v220, v250, v251
	v_cvt_pk_f16_f32 v221, v252, v253
	v_mfma_f32_16x16x32_f16 v[4:7], a[16:19], v[142:145], v[66:69]
	v_mfma_f32_16x16x32_f16 v[0:3], a[52:55], v[114:117], v[0:3]
	v_cvt_pk_f16_f32 v236, v110, v111
	v_cvt_pk_f16_f32 v237, v112, v113
	v_mfma_f32_16x16x32_f16 v[114:117], a[20:23], v[146:149], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[24:27], v[142:145], v[70:73]
	s_nop 3
	v_cvt_pk_f16_f32 v224, v0, v1
	v_cvt_pk_f16_f32 v225, v2, v3
	v_mfma_f32_16x16x32_f16 v[118:121], a[28:31], v[146:149], v[4:7]
	v_cvt_pk_f16_f32 v110, v114, v115
	v_cvt_pk_f16_f32 v111, v116, v117
	v_mfma_f32_16x16x32_f16 v[4:7], a[32:35], v[142:145], v[74:77]
	v_mfma_f32_16x16x32_f16 v[122:125], a[36:39], v[146:149], v[4:7]
	s_nop 3
	v_cvt_pk_f16_f32 v114, v118, v119
	v_cvt_pk_f16_f32 v115, v120, v121
	v_mfma_f32_16x16x32_f16 v[4:7], a[40:43], v[142:145], v[78:81]
	v_mfma_f32_16x16x32_f16 v[126:129], a[44:47], v[146:149], v[4:7]
	v_cvt_pk_f16_f32 v118, v122, v123
	v_cvt_pk_f16_f32 v119, v124, v125
	v_mfma_f32_16x16x32_f16 v[4:7], a[48:51], v[142:145], v[82:85]
	v_mfma_f32_16x16x32_f16 v[130:133], a[52:55], v[146:149], v[4:7]
	s_nop 3
	v_cvt_pk_f16_f32 v122, v126, v127
	v_cvt_pk_f16_f32 v123, v128, v129
	v_mfma_f32_16x16x32_f16 v[4:7], a[56:59], v[142:145], v[86:89]
	v_mfma_f32_16x16x32_f16 v[134:137], a[60:63], v[146:149], v[4:7]
	v_cvt_pk_f16_f32 v126, v130, v131
	v_cvt_pk_f16_f32 v127, v132, v133
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], v[142:145], v[90:93]
	v_mfma_f32_16x16x32_f16 v[138:141], a[68:71], v[146:149], v[4:7]
	s_nop 3
	v_cvt_pk_f16_f32 v130, v134, v135
	v_cvt_pk_f16_f32 v131, v136, v137
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], v[142:145], v[94:97]
	v_mfma_f32_16x16x32_f16 v[142:145], a[76:79], v[146:149], v[4:7]
	v_cvt_pk_f16_f32 v134, v138, v139
	v_cvt_pk_f16_f32 v135, v140, v141
	v_mfma_f32_16x16x32_f16 v[4:7], a[16:19], v[174:177], v[98:101]
	v_mfma_f32_16x16x32_f16 v[146:149], a[20:23], v[178:181], v[4:7]
	s_nop 3
	v_cvt_pk_f16_f32 v139, v144, v145
	v_cvt_pk_f16_f32 v138, v142, v143
	s_nop 0
	v_accvgpr_read_b32 v4, a104
	v_accvgpr_read_b32 v5, a105
	v_accvgpr_read_b32 v6, a106
	v_accvgpr_read_b32 v7, a107
	v_cvt_pk_f16_f32 v112, v146, v147
	v_cvt_pk_f16_f32 v113, v148, v149
	v_mfma_f32_16x16x32_f16 v[4:7], a[24:27], v[174:177], v[4:7]
	v_mfma_f32_16x16x32_f16 v[150:153], a[28:31], v[178:181], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a108
	v_accvgpr_read_b32 v5, a109
	v_accvgpr_read_b32 v6, a110
	v_accvgpr_read_b32 v7, a111
	v_cvt_pk_f16_f32 v116, v150, v151
	v_cvt_pk_f16_f32 v117, v152, v153
	v_mfma_f32_16x16x32_f16 v[4:7], a[32:35], v[174:177], v[4:7]
	v_mfma_f32_16x16x32_f16 v[154:157], a[36:39], v[178:181], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a112
	v_accvgpr_read_b32 v5, a113
	v_accvgpr_read_b32 v6, a114
	v_accvgpr_read_b32 v7, a115
	v_cvt_pk_f16_f32 v120, v154, v155
	v_cvt_pk_f16_f32 v121, v156, v157
	v_mfma_f32_16x16x32_f16 v[4:7], a[40:43], v[174:177], v[4:7]
	v_mfma_f32_16x16x32_f16 v[158:161], a[44:47], v[178:181], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a116
	v_accvgpr_read_b32 v5, a117
	v_accvgpr_read_b32 v6, a118
	v_accvgpr_read_b32 v7, a119
	v_cvt_pk_f16_f32 v124, v158, v159
	v_cvt_pk_f16_f32 v125, v160, v161
	v_mfma_f32_16x16x32_f16 v[4:7], a[48:51], v[174:177], v[4:7]
	v_mfma_f32_16x16x32_f16 v[162:165], a[52:55], v[178:181], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a120
	v_accvgpr_read_b32 v5, a121
	v_accvgpr_read_b32 v6, a122
	v_accvgpr_read_b32 v7, a123
	v_cvt_pk_f16_f32 v128, v162, v163
	v_cvt_pk_f16_f32 v129, v164, v165
	v_mfma_f32_16x16x32_f16 v[4:7], a[56:59], v[174:177], v[4:7]
	v_mfma_f32_16x16x32_f16 v[166:169], a[60:63], v[178:181], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a124
	v_accvgpr_read_b32 v5, a125
	v_accvgpr_read_b32 v6, a126
	v_accvgpr_read_b32 v7, a127
	v_cvt_pk_f16_f32 v132, v166, v167
	v_cvt_pk_f16_f32 v133, v168, v169
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], v[174:177], v[4:7]
	v_mfma_f32_16x16x32_f16 v[170:173], a[68:71], v[178:181], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a96
	v_accvgpr_read_b32 v5, a97
	v_accvgpr_read_b32 v6, a98
	v_accvgpr_read_b32 v7, a99
	v_cvt_pk_f16_f32 v136, v170, v171
	v_cvt_pk_f16_f32 v137, v172, v173
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], v[174:177], v[4:7]
	v_mfma_f32_16x16x32_f16 v[174:177], a[76:79], v[178:181], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a100
	v_accvgpr_read_b32 v5, a101
	v_accvgpr_read_b32 v6, a102
	v_accvgpr_read_b32 v7, a103
	v_cvt_pk_f16_f32 v140, v174, v175
	v_cvt_pk_f16_f32 v141, v176, v177
	v_mfma_f32_16x16x32_f16 v[4:7], a[16:19], a[88:91], v[4:7]
	v_mfma_f32_16x16x32_f16 v[178:181], a[20:23], a[92:95], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a136
	v_accvgpr_read_b32 v5, a137
	v_accvgpr_read_b32 v6, a138
	v_accvgpr_read_b32 v7, a139
	v_cvt_pk_f16_f32 v142, v178, v179
	v_cvt_pk_f16_f32 v143, v180, v181
	v_mfma_f32_16x16x32_f16 v[4:7], a[24:27], a[88:91], v[4:7]
	v_mfma_f32_16x16x32_f16 v[182:185], a[28:31], a[92:95], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a140
	v_accvgpr_read_b32 v5, a141
	v_accvgpr_read_b32 v6, a142
	v_accvgpr_read_b32 v7, a143
	v_cvt_pk_f16_f32 v146, v182, v183
	v_cvt_pk_f16_f32 v147, v184, v185
	v_mfma_f32_16x16x32_f16 v[4:7], a[32:35], a[88:91], v[4:7]
	v_mfma_f32_16x16x32_f16 v[186:189], a[36:39], a[92:95], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a144
	v_accvgpr_read_b32 v5, a145
	v_accvgpr_read_b32 v6, a146
	v_accvgpr_read_b32 v7, a147
	v_cvt_pk_f16_f32 v150, v186, v187
	v_cvt_pk_f16_f32 v151, v188, v189
	v_mfma_f32_16x16x32_f16 v[4:7], a[40:43], a[88:91], v[4:7]
	v_mfma_f32_16x16x32_f16 v[190:193], a[44:47], a[92:95], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a160
	v_accvgpr_read_b32 v5, a161
	v_accvgpr_read_b32 v6, a162
	v_accvgpr_read_b32 v7, a163
	v_cvt_pk_f16_f32 v102, v190, v191
	v_cvt_pk_f16_f32 v103, v192, v193
	v_mfma_f32_16x16x32_f16 v[4:7], a[48:51], a[88:91], v[4:7]
	v_mfma_f32_16x16x32_f16 v[194:197], a[52:55], a[92:95], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a164
	v_accvgpr_read_b32 v5, a165
	v_accvgpr_read_b32 v6, a166
	v_accvgpr_read_b32 v7, a167
	v_cvt_pk_f16_f32 v154, v194, v195
	v_cvt_pk_f16_f32 v155, v196, v197
	v_mfma_f32_16x16x32_f16 v[4:7], a[56:59], a[88:91], v[4:7]
	v_mfma_f32_16x16x32_f16 v[198:201], a[60:63], a[92:95], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a168
	v_accvgpr_read_b32 v5, a169
	v_accvgpr_read_b32 v6, a170
	v_accvgpr_read_b32 v7, a171
	v_cvt_pk_f16_f32 v158, v198, v199
	v_cvt_pk_f16_f32 v159, v200, v201
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], a[88:91], v[4:7]
	v_mfma_f32_16x16x32_f16 v[202:205], a[68:71], a[92:95], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a172
	v_accvgpr_read_b32 v5, a173
	v_accvgpr_read_b32 v6, a174
	v_accvgpr_read_b32 v7, a175
	v_cvt_pk_f16_f32 v162, v202, v203
	v_cvt_pk_f16_f32 v163, v204, v205
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], a[88:91], v[4:7]
	v_mfma_f32_16x16x32_f16 v[66:69], a[76:79], a[92:95], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a176
	v_accvgpr_read_b32 v5, a177
	v_accvgpr_read_b32 v6, a178
	v_accvgpr_read_b32 v7, a179
	v_cvt_pk_f16_f32 v106, v66, v67
	v_cvt_pk_f16_f32 v107, v68, v69
	v_mfma_f32_16x16x32_f16 v[4:7], a[16:19], a[80:83], v[4:7]
	v_mfma_f32_16x16x32_f16 v[70:73], a[20:23], a[84:87], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a180
	v_accvgpr_read_b32 v5, a181
	v_accvgpr_read_b32 v6, a182
	v_accvgpr_read_b32 v7, a183
	v_cvt_pk_f16_f32 v144, v70, v71
	v_cvt_pk_f16_f32 v145, v72, v73
	v_mfma_f32_16x16x32_f16 v[4:7], a[24:27], a[80:83], v[4:7]
	v_mfma_f32_16x16x32_f16 v[74:77], a[28:31], a[84:87], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a184
	v_accvgpr_read_b32 v5, a185
	v_accvgpr_read_b32 v6, a186
	v_accvgpr_read_b32 v7, a187
	v_cvt_pk_f16_f32 v148, v74, v75
	v_cvt_pk_f16_f32 v149, v76, v77
	v_mfma_f32_16x16x32_f16 v[4:7], a[32:35], a[80:83], v[4:7]
	v_mfma_f32_16x16x32_f16 v[78:81], a[36:39], a[84:87], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a188
	v_accvgpr_read_b32 v5, a189
	v_accvgpr_read_b32 v6, a190
	v_accvgpr_read_b32 v7, a191
	v_cvt_pk_f16_f32 v152, v78, v79
	v_cvt_pk_f16_f32 v153, v80, v81
	v_mfma_f32_16x16x32_f16 v[4:7], a[40:43], a[80:83], v[4:7]
	v_mfma_f32_16x16x32_f16 v[82:85], a[44:47], a[84:87], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a192
	v_accvgpr_read_b32 v5, a193
	v_accvgpr_read_b32 v6, a194
	v_accvgpr_read_b32 v7, a195
	v_cvt_pk_f16_f32 v104, v82, v83
	v_cvt_pk_f16_f32 v105, v84, v85
	v_mfma_f32_16x16x32_f16 v[4:7], a[48:51], a[80:83], v[4:7]
	v_mfma_f32_16x16x32_f16 v[86:89], a[52:55], a[84:87], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a196
	v_accvgpr_read_b32 v5, a197
	v_accvgpr_read_b32 v6, a198
	v_accvgpr_read_b32 v7, a199
	v_cvt_pk_f16_f32 v156, v86, v87
	v_cvt_pk_f16_f32 v157, v88, v89
	v_mfma_f32_16x16x32_f16 v[4:7], a[56:59], a[80:83], v[4:7]
	v_mfma_f32_16x16x32_f16 v[90:93], a[60:63], a[84:87], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a200
	v_accvgpr_read_b32 v5, a201
	v_accvgpr_read_b32 v6, a202
	v_accvgpr_read_b32 v7, a203
	v_cvt_pk_f16_f32 v160, v90, v91
	v_cvt_pk_f16_f32 v161, v92, v93
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], a[80:83], v[4:7]
	v_mfma_f32_16x16x32_f16 v[94:97], a[68:71], a[84:87], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a204
	v_accvgpr_read_b32 v5, a205
	v_accvgpr_read_b32 v6, a206
	v_accvgpr_read_b32 v7, a207
	v_cvt_pk_f16_f32 v164, v94, v95
	v_cvt_pk_f16_f32 v165, v96, v97
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], a[80:83], v[4:7]
	v_mfma_f32_16x16x32_f16 v[98:101], a[76:79], a[84:87], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a208
	v_accvgpr_read_b32 v5, a209
	v_accvgpr_read_b32 v6, a210
	v_accvgpr_read_b32 v7, a211
	v_cvt_pk_f16_f32 v108, v98, v99
	v_cvt_pk_f16_f32 v109, v100, v101
	v_mfma_f32_16x16x32_f16 v[4:7], a[16:19], a[8:11], v[4:7]
	v_mfma_f32_16x16x32_f16 v[38:41], a[20:23], a[12:15], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a212
	v_accvgpr_read_b32 v5, a213
	v_accvgpr_read_b32 v6, a214
	v_accvgpr_read_b32 v7, a215
	v_cvt_pk_f16_f32 v70, v38, v39
	v_cvt_pk_f16_f32 v71, v40, v41
	v_mfma_f32_16x16x32_f16 v[4:7], a[24:27], a[8:11], v[4:7]
	v_mfma_f32_16x16x32_f16 v[42:45], a[28:31], a[12:15], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a216
	v_accvgpr_read_b32 v5, a217
	v_accvgpr_read_b32 v6, a218
	v_accvgpr_read_b32 v7, a219
	v_cvt_pk_f16_f32 v67, v44, v45
	v_cvt_pk_f16_f32 v66, v42, v43
	v_mfma_f32_16x16x32_f16 v[4:7], a[32:35], a[8:11], v[4:7]
	v_mfma_f32_16x16x32_f16 v[46:49], a[36:39], a[12:15], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a220
	v_accvgpr_read_b32 v5, a221
	v_accvgpr_read_b32 v6, a222
	v_accvgpr_read_b32 v7, a223
	v_cvt_pk_f16_f32 v39, v48, v49
	v_cvt_pk_f16_f32 v38, v46, v47
	v_mfma_f32_16x16x32_f16 v[4:7], a[40:43], a[8:11], v[4:7]
	v_mfma_f32_16x16x32_f16 v[50:53], a[44:47], a[12:15], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a224
	v_accvgpr_read_b32 v5, a225
	v_accvgpr_read_b32 v6, a226
	v_accvgpr_read_b32 v7, a227
	v_cvt_pk_f16_f32 v1, v52, v53
	v_cvt_pk_f16_f32 v0, v50, v51
	v_mfma_f32_16x16x32_f16 v[4:7], a[48:51], a[8:11], v[4:7]
	v_mfma_f32_16x16x32_f16 v[54:57], a[52:55], a[12:15], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a228
	v_accvgpr_read_b32 v5, a229
	v_accvgpr_read_b32 v6, a230
	v_accvgpr_read_b32 v7, a231
	v_cvt_pk_f16_f32 v50, v54, v55
	v_cvt_pk_f16_f32 v51, v56, v57
	v_mfma_f32_16x16x32_f16 v[4:7], a[56:59], a[8:11], v[4:7]
	v_mfma_f32_16x16x32_f16 v[58:61], a[60:63], a[12:15], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a232
	v_accvgpr_read_b32 v5, a233
	v_accvgpr_read_b32 v6, a234
	v_accvgpr_read_b32 v7, a235
	v_cvt_pk_f16_f32 v46, v58, v59
	v_cvt_pk_f16_f32 v47, v60, v61
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], a[8:11], v[4:7]
	v_mfma_f32_16x16x32_f16 v[62:65], a[68:71], a[12:15], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a236
	v_accvgpr_read_b32 v5, a237
	v_accvgpr_read_b32 v6, a238
	v_accvgpr_read_b32 v7, a239
	v_cvt_pk_f16_f32 v42, v62, v63
	v_cvt_pk_f16_f32 v43, v64, v65
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], a[8:11], v[4:7]
	v_accvgpr_mov_b32 a8, a148
	v_accvgpr_mov_b32 a9, a149
	v_accvgpr_mov_b32 a10, a150
	v_mfma_f32_16x16x32_f16 v[34:37], a[76:79], a[12:15], v[4:7]
	v_accvgpr_mov_b32 a11, a151
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[8:11], a[56:59], a[0:3], a[8:11]
	v_accvgpr_read_b32 v4, a240
	v_accvgpr_read_b32 v5, a241
	v_accvgpr_read_b32 v6, a242
	v_accvgpr_read_b32 v7, a243
	v_mfma_f32_16x16x32_f16 a[8:11], a[60:63], a[4:7], a[8:11]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[4:7], a[16:19], a[0:3], v[4:7]
	v_mfma_f32_16x16x32_f16 v[22:25], a[20:23], a[4:7], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a244
	v_accvgpr_read_b32 v5, a245
	v_accvgpr_read_b32 v6, a246
	v_accvgpr_read_b32 v7, a247
	v_cvt_pk_f16_f32 v72, v22, v23
	v_cvt_pk_f16_f32 v73, v24, v25
	v_mfma_f32_16x16x32_f16 v[4:7], a[24:27], a[0:3], v[4:7]
	v_mfma_f32_16x16x32_f16 v[26:29], a[28:31], a[4:7], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a248
	v_accvgpr_read_b32 v5, a249
	v_accvgpr_read_b32 v6, a250
	v_accvgpr_read_b32 v7, a251
	v_cvt_pk_f16_f32 v68, v26, v27
	v_cvt_pk_f16_f32 v69, v28, v29
	v_mfma_f32_16x16x32_f16 v[4:7], a[32:35], a[0:3], v[4:7]
	v_mfma_f32_16x16x32_f16 v[30:33], a[36:39], a[4:7], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a252
	v_accvgpr_read_b32 v5, a253
	v_accvgpr_read_b32 v6, a254
	v_accvgpr_read_b32 v7, a255
	v_cvt_pk_f16_f32 v41, v32, v33
	scratch_load_dword v32, off, off offset:4 ; 4-byte Folded Reload
	v_mfma_f32_16x16x32_f16 v[4:7], a[40:43], a[0:3], v[4:7]
	v_cvt_pk_f16_f32 v40, v30, v31
	v_mfma_f32_16x16x32_f16 v[18:21], a[44:47], a[4:7], v[4:7]
	s_nop 5
	v_accvgpr_read_b32 v4, a128
	v_accvgpr_read_b32 v5, a129
	v_accvgpr_read_b32 v6, a130
	v_accvgpr_read_b32 v7, a131
	v_cvt_pk_f16_f32 v2, v18, v19
	v_cvt_pk_f16_f32 v3, v20, v21
	v_mfma_f32_16x16x32_f16 v[4:7], a[48:51], a[0:3], v[4:7]
	v_mfma_f32_16x16x32_f16 v[14:17], a[52:55], a[4:7], v[4:7]
	s_nop 6
	v_accvgpr_read_b32 v4, a152
	v_accvgpr_read_b32 v5, a153
	v_accvgpr_read_b32 v6, a154
	v_accvgpr_read_b32 v7, a155
	v_cvt_pk_f16_f32 v52, v14, v15
	v_cvt_pk_f16_f32 v53, v16, v17
	v_accvgpr_read_b32 v17, a11
	v_accvgpr_read_b32 v16, a10
	v_cvt_pk_f16_f32 v49, v16, v17
	scratch_load_dword v16, off, off offset:8 ; 4-byte Folded Reload
	v_mfma_f32_16x16x32_f16 v[10:13], a[64:67], a[0:3], v[4:7]
	v_accvgpr_read_b32 v15, a9
	v_accvgpr_read_b32 v14, a8
	v_cvt_pk_f16_f32 v48, v14, v15
	v_accvgpr_read_b32 v4, a156
	v_accvgpr_read_b32 v5, a157
	v_accvgpr_read_b32 v6, a158
	v_accvgpr_read_b32 v7, a159
	v_mfma_f32_16x16x32_f16 v[10:13], a[68:71], a[4:7], v[10:13]
	s_nop 0
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], a[0:3], v[4:7]
	v_mfma_f32_16x16x32_f16 v[6:9], a[76:79], a[4:7], v[4:7]
	s_nop 4
	v_cvt_pk_f16_f32 v44, v10, v11
	v_cvt_pk_f16_f32 v45, v12, v13
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v10, 1, v16
	v_cvt_pk_f16_f32 v6, v6, v7
	v_cvt_pk_f16_f32 v7, v8, v9
	v_lshlrev_b32_e32 v8, 9, v16
	v_and_b32_e32 v12, 16, v16
	v_and_b32_e32 v8, 0x5c00, v8
	v_and_b32_e32 v9, 0x70, v32
	v_lshlrev_b32_e32 v11, 13, v10
	v_lshlrev_b32_e32 v13, 4, v12
	v_or3_b32 v11, v11, v13, s1
	v_bitop3_b32 v8, s0, v8, v9 bitop3:0x1e
	s_movk_i32 s0, 0x60
	v_or_b32_e32 v13, v8, v11
	v_bitop3_b32 v8, v8, s0, v11 bitop3:0x36
	v_lshlrev_b32_e32 v11, 5, v12
	v_lshlrev_b32_e32 v12, 8, v16
	s_lshl_b32 s1, s15, 4
	v_lshlrev_b32_e32 v10, 14, v10
	v_and_b32_e32 v12, 0x2000, v12
	s_lshl_b32 s0, s15, 10
	v_or3_b32 v10, v10, v11, v12
	v_mov_b32_e32 v11, s1
	v_bitop3_b32 v9, s0, v11, v9 bitop3:0x36
	s_movk_i32 s0, 0x1040
	v_or_b32_e32 v11, v9, v10
	v_bitop3_b32 v9, v9, s0, v10 bitop3:0x36
	v_add_u32_e32 v14, 0, v13
	v_xad_u32 v15, v13, 32, 0
	v_xad_u32 v13, v13, 64, 0
	v_add_u32_e32 v8, 0, v8
	v_add_u32_e32 v11, 0, v11
	v_add_u32_e32 v9, 0, v9
	v_cvt_pk_f16_f32 v4, v34, v35
	v_cvt_pk_f16_f32 v5, v36, v37
	ds_write_b128 v14, v[206:209]
	ds_write_b128 v14, v[222:225] offset:512
	ds_write_b128 v15, v[210:213]
	ds_write_b128 v15, v[226:229] offset:512
	ds_write_b128 v13, v[214:217]
	ds_write_b128 v13, v[230:233] offset:512
	ds_write_b128 v8, v[218:221]
	ds_write_b128 v8, v[234:237] offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[54:57], v11
	ds_read_b128 v[60:63], v11 offset:256
	ds_read_b128 v[74:77], v11 offset:128
	ds_read_b128 v[80:83], v11 offset:384
	ds_read_b128 v[84:87], v9
	ds_read_b128 v[90:93], v9 offset:256
	ds_read_b128 v[94:97], v9 offset:128
	ds_read_b128 v[166:169], v9 offset:384
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v14, v[110:113]
	ds_write_b128 v14, v[126:129] offset:512
	ds_write_b128 v15, v[114:117]
	ds_write_b128 v15, v[130:133] offset:512
	ds_write_b128 v13, v[118:121]
	ds_write_b128 v13, v[134:137] offset:512
	ds_write_b128 v8, v[122:125]
	ds_write_b128 v8, v[138:141] offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[110:113], v11
	ds_read_b128 v[116:119], v11 offset:256
	ds_read_b128 v[120:123], v11 offset:128
	ds_read_b128 v[126:129], v11 offset:384
	ds_read_b128 v[130:133], v9
	ds_read_b128 v[136:139], v9 offset:256
	ds_read_b128 v[170:173], v9 offset:128
	ds_read_b128 v[176:179], v9 offset:384
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v14, v[142:145]
	ds_write_b128 v14, v[154:157] offset:512
	ds_write_b128 v15, v[146:149]
	ds_write_b128 v15, v[158:161] offset:512
	ds_write_b128 v13, v[150:153]
	ds_write_b128 v13, v[162:165] offset:512
	ds_write_b128 v8, v[102:105]
	ds_write_b128 v8, v[106:109] offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[100:103], v11
	ds_read_b128 v[106:109], v11 offset:256
	ds_read_b128 v[140:143], v11 offset:128
	ds_read_b128 v[146:149], v11 offset:384
	ds_read_b128 v[150:153], v9
	ds_read_b128 v[156:159], v9 offset:256
	ds_read_b128 v[160:163], v9 offset:128
	ds_read_b128 v[180:183], v9 offset:384
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v14, v[70:73]
	ds_write_b128 v14, v[50:53] offset:512
	ds_write_b128 v15, v[66:69]
	ds_write_b128 v15, v[46:49] offset:512
	ds_write_b128 v13, v[38:41]
	ds_write_b128 v13, v[42:45] offset:512
	ds_write_b128 v8, v[0:3]
	ds_write_b128 v8, v[4:7] offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	scratch_load_dword v0, off, off         ; 4-byte Folded Reload
	s_mul_i32 s0, s13, s12
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s6, s0
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s15, s14, 31
	v_and_b32_e32 v32, 0xf8, v32
	s_lshl_b64 s[0:1], s[14:15], 1
	s_add_u32 s52, s2, s0
	v_cmp_gt_i32_e64 s[42:43], s9, v32
	ds_read_b128 v[50:53], v11
	ds_read_b128 v[64:67], v11 offset:256
	ds_read_b128 v[68:71], v11 offset:128
	ds_read_b128 v[184:187], v11 offset:384
	ds_read_b128 v[188:191], v9
	ds_read_b128 v[194:197], v9 offset:256
	ds_read_b128 v[198:201], v9 offset:128
	ds_read_b128 v[204:207], v9 offset:384
	s_addc_u32 s33, s3, s1
	s_and_b32 s53, s33, 0xffff
	v_mov_b32_e32 v58, v54
	v_mov_b32_e32 v59, v55
	v_mov_b32_e32 v88, v84
	v_mov_b32_e32 v89, v85
	v_mov_b32_e32 v78, v74
	v_mov_b32_e32 v79, v75
	v_mov_b32_e32 v164, v94
	v_mov_b32_e32 v165, v95
	v_mov_b32_e32 v98, v168
	v_mov_b32_e32 v99, v169
	v_mov_b32_e32 v114, v110
	v_mov_b32_e32 v115, v111
	v_mov_b32_e32 v134, v130
	v_mov_b32_e32 v135, v131
	v_mov_b32_e32 v124, v120
	v_mov_b32_e32 v125, v121
	v_mov_b32_e32 v174, v170
	v_mov_b32_e32 v175, v171
	v_mov_b32_e32 v104, v100
	v_mov_b32_e32 v105, v101
	v_mov_b32_e32 v154, v150
	v_mov_b32_e32 v155, v151
	v_mov_b32_e32 v144, v140
	v_mov_b32_e32 v145, v141
	s_waitcnt lgkmcnt(3)
	v_mov_b32_e32 v192, v188
	v_mov_b32_e32 v193, v189
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v202, v198
	v_mov_b32_e32 v203, v199
	v_mov_b32_e32 v54, v66
	v_mov_b32_e32 v55, v67
	v_mov_b32_e32 v72, v186
	v_mov_b32_e32 v73, v187
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v0, 5, v0
	v_or_b32_e32 v2, 16, v0
	v_mul_lo_u32 v33, v0, s12
	v_cmp_gt_i32_e32 vcc, s8, v0
	v_or_b32_e32 v1, 8, v0
	v_or_b32_e32 v3, 24, v0
	v_or_b32_e32 v4, 32, v0
	v_or_b32_e32 v5, 40, v0
	v_or_b32_e32 v6, 48, v0
	v_or_b32_e32 v7, 56, v0
	v_or_b32_e32 v8, 64, v0
	v_or_b32_e32 v9, 0x48, v0
	v_or_b32_e32 v10, 0x50, v0
	v_or_b32_e32 v11, 0x58, v0
	v_or_b32_e32 v12, 0x60, v0
	v_or_b32_e32 v13, 0x68, v0
	v_or_b32_e32 v14, 0x70, v0
	v_or_b32_e32 v15, 0x78, v0
	v_or_b32_e32 v16, 0x80, v0
	v_or_b32_e32 v17, 0x88, v0
	v_or_b32_e32 v18, 0x90, v0
	v_or_b32_e32 v19, 0x98, v0
	v_or_b32_e32 v20, 0xa0, v0
	v_or_b32_e32 v21, 0xa8, v0
	v_or_b32_e32 v22, 0xb0, v0
	v_or_b32_e32 v23, 0xb8, v0
	v_or_b32_e32 v24, 0xc0, v0
	v_or_b32_e32 v25, 0xc8, v0
	v_or_b32_e32 v26, 0xd0, v0
	v_or_b32_e32 v27, 0xd8, v0
	v_or_b32_e32 v28, 0xe0, v0
	v_or_b32_e32 v29, 0xe8, v0
	v_or_b32_e32 v30, 0xf0, v0
	v_or_b32_e32 v31, 0xf8, v0
	v_mul_lo_u32 v34, v2, s12
	v_cmp_gt_i32_e64 s[0:1], s8, v2
	s_and_b64 s[44:45], s[42:43], vcc
	v_add_lshl_u32 v0, v33, v32, 1
	v_bfrev_b32_e32 v2, 1
	v_mul_lo_u32 v1, v1, s12
	v_cndmask_b32_e64 v0, v2, v0, s[44:45]
	buffer_store_dwordx4 v[58:61], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v1, v32, 1
	v_cndmask_b32_e64 v0, v2, v0, s[44:45]
	s_and_b64 s[46:47], s[42:43], s[0:1]
	buffer_store_dwordx4 v[88:91], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v34, v32, 1
	v_mul_lo_u32 v3, v3, s12
	v_cndmask_b32_e64 v0, v2, v0, s[46:47]
	buffer_store_dwordx4 v[78:81], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v3, v32, 1
	v_mul_lo_u32 v35, v4, s12
	v_cmp_gt_i32_e64 s[2:3], s8, v4
	v_cndmask_b32_e64 v0, v2, v0, s[46:47]
	s_and_b64 s[48:49], s[42:43], s[2:3]
	buffer_store_dwordx4 v[164:167], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v35, v32, 1
	v_mul_lo_u32 v5, v5, s12
	v_mov_b32_e32 v58, v62
	v_mov_b32_e32 v59, v63
	v_cndmask_b32_e64 v0, v2, v0, s[48:49]
	buffer_store_dwordx4 v[56:59], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v5, v32, 1
	v_mul_lo_u32 v36, v6, s12
	v_cmp_gt_i32_e64 s[4:5], s8, v6
	v_mov_b32_e32 v88, v92
	v_mov_b32_e32 v89, v93
	v_cndmask_b32_e64 v0, v2, v0, s[48:49]
	s_and_b64 s[22:23], s[42:43], s[4:5]
	buffer_store_dwordx4 v[86:89], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v36, v32, 1
	v_mul_lo_u32 v7, v7, s12
	v_mov_b32_e32 v78, v82
	v_mov_b32_e32 v79, v83
	v_cndmask_b32_e64 v0, v2, v0, s[22:23]
	buffer_store_dwordx4 v[76:79], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v7, v32, 1
	v_mul_lo_u32 v37, v8, s12
	v_cmp_gt_i32_e64 s[6:7], s8, v8
	v_cndmask_b32_e64 v0, v2, v0, s[22:23]
	s_and_b64 s[20:21], s[42:43], s[6:7]
	buffer_store_dwordx4 v[96:99], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v37, v32, 1
	v_mul_lo_u32 v9, v9, s12
	v_cndmask_b32_e64 v0, v2, v0, s[20:21]
	buffer_store_dwordx4 v[114:117], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v9, v32, 1
	v_mul_lo_u32 v38, v10, s12
	v_cmp_gt_i32_e64 s[10:11], s8, v10
	v_cndmask_b32_e64 v0, v2, v0, s[20:21]
	s_and_b64 s[18:19], s[42:43], s[10:11]
	buffer_store_dwordx4 v[134:137], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v38, v32, 1
	v_mul_lo_u32 v11, v11, s12
	v_cndmask_b32_e64 v0, v2, v0, s[18:19]
	buffer_store_dwordx4 v[124:127], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v11, v32, 1
	v_mul_lo_u32 v39, v12, s12
	v_mul_lo_u32 v13, v13, s12
	v_mul_lo_u32 v40, v14, s12
	v_mul_lo_u32 v15, v15, s12
	v_mul_lo_u32 v41, v16, s12
	v_mul_lo_u32 v17, v17, s12
	v_mul_lo_u32 v42, v18, s12
	v_mul_lo_u32 v19, v19, s12
	v_mul_lo_u32 v43, v20, s12
	v_mul_lo_u32 v21, v21, s12
	v_mul_lo_u32 v44, v22, s12
	v_mul_lo_u32 v23, v23, s12
	v_mul_lo_u32 v45, v24, s12
	v_mul_lo_u32 v25, v25, s12
	v_mul_lo_u32 v46, v26, s12
	v_mul_lo_u32 v27, v27, s12
	v_mul_lo_u32 v47, v28, s12
	v_mul_lo_u32 v29, v29, s12
	v_mul_lo_u32 v48, v30, s12
	v_mul_lo_u32 v31, v31, s12
	v_cmp_gt_i32_e64 s[12:13], s8, v12
	v_cndmask_b32_e64 v0, v2, v0, s[18:19]
	s_and_b64 s[16:17], s[42:43], s[12:13]
	buffer_store_dwordx4 v[174:177], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v39, v32, 1
	v_mov_b32_e32 v114, v118
	v_mov_b32_e32 v115, v119
	v_cndmask_b32_e64 v0, v2, v0, s[16:17]
	buffer_store_dwordx4 v[112:115], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v13, v32, 1
	v_cmp_gt_i32_e64 s[14:15], s8, v14
	v_mov_b32_e32 v134, v138
	v_mov_b32_e32 v135, v139
	v_cndmask_b32_e64 v0, v2, v0, s[16:17]
	s_and_b64 s[14:15], s[42:43], s[14:15]
	buffer_store_dwordx4 v[132:135], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v40, v32, 1
	v_mov_b32_e32 v124, v128
	v_mov_b32_e32 v125, v129
	v_cndmask_b32_e64 v0, v2, v0, s[14:15]
	buffer_store_dwordx4 v[122:125], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v15, v32, 1
	v_cmp_gt_i32_e64 s[24:25], s8, v16
	v_mov_b32_e32 v174, v178
	v_mov_b32_e32 v175, v179
	v_cndmask_b32_e64 v0, v2, v0, s[14:15]
	s_and_b64 s[12:13], s[42:43], s[24:25]
	buffer_store_dwordx4 v[172:175], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v41, v32, 1
	v_cndmask_b32_e64 v0, v2, v0, s[12:13]
	buffer_store_dwordx4 v[104:107], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v17, v32, 1
	v_cmp_gt_i32_e64 s[26:27], s8, v18
	v_cndmask_b32_e64 v0, v2, v0, s[12:13]
	s_and_b64 s[10:11], s[42:43], s[26:27]
	buffer_store_dwordx4 v[154:157], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v42, v32, 1
	v_cndmask_b32_e64 v0, v2, v0, s[10:11]
	buffer_store_dwordx4 v[144:147], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v19, v32, 1
	v_cmp_gt_i32_e64 s[28:29], s8, v20
	v_mov_b32_e32 v178, v160
	v_mov_b32_e32 v179, v161
	v_cndmask_b32_e64 v0, v2, v0, s[10:11]
	v_cmp_gt_i32_e64 s[30:31], s8, v22
	v_cmp_gt_i32_e64 s[34:35], s8, v24
	v_cmp_gt_i32_e64 s[36:37], s8, v26
	v_cmp_gt_i32_e64 s[38:39], s8, v28
	v_cmp_gt_i32_e64 s[40:41], s8, v30
	s_and_b64 s[8:9], s[42:43], s[28:29]
	buffer_store_dwordx4 v[178:181], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v43, v32, 1
	v_mov_b32_e32 v104, v108
	v_mov_b32_e32 v105, v109
	v_cndmask_b32_e64 v0, v2, v0, s[8:9]
	buffer_store_dwordx4 v[102:105], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v21, v32, 1
	v_mov_b32_e32 v154, v158
	v_mov_b32_e32 v155, v159
	v_cndmask_b32_e64 v0, v2, v0, s[8:9]
	s_and_b64 s[6:7], s[42:43], s[30:31]
	buffer_store_dwordx4 v[152:155], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v44, v32, 1
	v_mov_b32_e32 v144, v148
	v_mov_b32_e32 v145, v149
	v_cndmask_b32_e64 v0, v2, v0, s[6:7]
	buffer_store_dwordx4 v[142:145], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v23, v32, 1
	v_mov_b32_e32 v164, v182
	v_mov_b32_e32 v165, v183
	v_cndmask_b32_e64 v0, v2, v0, s[6:7]
	s_and_b64 s[4:5], s[42:43], s[34:35]
	buffer_store_dwordx4 v[162:165], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v45, v32, 1
	v_mov_b32_e32 v62, v50
	v_mov_b32_e32 v63, v51
	v_cndmask_b32_e64 v0, v2, v0, s[4:5]
	buffer_store_dwordx4 v[62:65], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v25, v32, 1
	v_cndmask_b32_e64 v0, v2, v0, s[4:5]
	s_and_b64 s[2:3], s[42:43], s[36:37]
	buffer_store_dwordx4 v[192:195], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v46, v32, 1
	v_mov_b32_e32 v182, v68
	v_mov_b32_e32 v183, v69
	v_cndmask_b32_e64 v0, v2, v0, s[2:3]
	buffer_store_dwordx4 v[182:185], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v27, v32, 1
	v_cndmask_b32_e64 v0, v2, v0, s[2:3]
	s_and_b64 s[0:1], s[42:43], s[38:39]
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[202:205], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v47, v32, 1
	v_cndmask_b32_e64 v0, v2, v0, s[0:1]
	buffer_store_dwordx4 v[52:55], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v29, v32, 1
	v_mov_b32_e32 v192, v196
	v_mov_b32_e32 v193, v197
	v_cndmask_b32_e64 v0, v2, v0, s[0:1]
	s_and_b64 vcc, s[42:43], s[40:41]
	buffer_store_dwordx4 v[190:193], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v48, v32, 1
	v_cndmask_b32_e32 v0, v2, v0, vcc
	buffer_store_dwordx4 v[70:73], v0, s[52:55], 0 offen
	v_add_lshl_u32 v0, v31, v32, 1
	v_mov_b32_e32 v202, v206
	v_mov_b32_e32 v203, v207
	v_cndmask_b32_e32 v0, v2, v0, vcc
	buffer_store_dwordx4 v[200:203], v0, s[52:55], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v6_loop_unroll
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 16
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 61
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
	.size	v6_loop_unroll, .Lfunc_end0-v6_loop_unroll
	.cfi_endproc
                                        ; -- End function
	.set v6_loop_unroll.num_vgpr, 256
	.set v6_loop_unroll.num_agpr, 256
	.set v6_loop_unroll.numbered_sgpr, 61
	.set v6_loop_unroll.num_named_barrier, 0
	.set v6_loop_unroll.private_seg_size, 16
	.set v6_loop_unroll.uses_vcc, 1
	.set v6_loop_unroll.uses_flat_scratch, 0
	.set v6_loop_unroll.has_dyn_sized_stack, 0
	.set v6_loop_unroll.has_recursion, 0
	.set v6_loop_unroll.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15328
; TotalNumSgprs: 67
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 16
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 8
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 67
; NumVGPRsForWavesPerEU: 512
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	.byte	1                               ; Abbrev [1] 0xb:0x4c DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x26 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	54                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7 ; matmul_kernel.py
.Linfo_string2:
	.asciz	"/var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v6_loop_unroll" ; string offset=24 ; /var/lib/jenkins/gfx9-gluon-tutorials/kernels/gemm/a16w16/v6_loop_unroll
.Linfo_string3:
	.asciz	"v6_loop_unroll"                ; string offset=97 ; v6_loop_unroll
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
    .name:           v6_loop_unroll
    .private_segment_fixed_size: 16
    .sgpr_count:     67
    .sgpr_spill_count: 0
    .symbol:         v6_loop_unroll.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
    .vgpr_spill_count: 3
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
