	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v5_local_prefetch               ; -- Begin function v5_local_prefetch
	.p2align	8
	.type	v5_local_prefetch,@function
v5_local_prefetch:                      ; @v5_local_prefetch
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.9:
	.file	1 "/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v5_local_prefetch" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.10:
.LBB0_0:
	s_mov_b32 s1, s9
	v_readfirstlane_b32 s9, v0
	s_bfe_u32 s24, s9, 0x20006
	.file	2 "/root/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s1, 0xff
	s_ashr_i32 s14, s0, 31
	s_lshr_b32 s14, s14, 24
	s_add_i32 s0, s0, s14
	s_ashr_i32 s0, s0, 8
	s_xor_b32 s14, s16, s0
	s_ashr_i32 s15, s14, 31
	s_abs_i32 s17, s16
	s_abs_i32 s18, s0
	v_cvt_f32_u32_e32 v1, s18
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_mov_b32 s14, 0
	s_sub_i32 s19, 0, s18
	v_readfirstlane_b32 s20, v1
	s_mul_i32 s19, s19, s20
	s_mul_hi_u32 s19, s20, s19
	s_add_i32 s20, s20, s19
	s_mul_hi_u32 s19, s17, s20
	s_mul_i32 s20, s19, s18
	s_sub_i32 s17, s17, s20
	s_add_i32 s20, s19, 1
	s_sub_i32 s21, s17, s18
	s_cmp_ge_u32 s17, s18
	s_cselect_b32 s19, s20, s19
	s_cselect_b32 s17, s21, s17
	s_add_i32 s20, s19, 1
	s_cmp_ge_u32 s17, s18
	s_cselect_b32 s17, s20, s19
	s_xor_b32 s17, s17, s15
	s_sub_i32 s15, s17, s15
	s_mul_i32 s0, s15, s0
	s_sub_i32 s18, s16, s0
	v_and_b32_e32 v1, 63, v0
	v_lshl_or_b32 v254, s24, 6, v1
	v_lshlrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v1, 0x70, v1
	v_or_b32_e32 v1, s24, v1
	v_lshlrev_b32_e32 v2, 3, v0
	v_and_b32_e32 v2, 56, v2
	s_lshl_b32 s0, s15, 8
	s_mul_i32 s16, s0, s11
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[16:17], s[16:17], 1
	s_add_u32 s20, s2, s16
	s_addc_u32 s15, s3, s17
	s_lshl_b32 s2, s18, 8
	s_mul_i32 s16, s2, s12
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[16:17], s[16:17], 1
	s_add_u32 s16, s4, s16
	s_addc_u32 s5, s5, s17
	s_lshl_b32 s3, s11, 2
	s_mul_i32 s4, s11, 0x74
	v_mad_u64_u32 v[4:5], s[18:19], v1, s11, v[2:3]
	v_add_u32_e32 v5, s3, v4
	v_add_u32_e32 v6, s3, v5
	v_add_u32_e32 v7, s3, v6
	v_add_u32_e32 v8, s4, v7
	v_add_u32_e32 v9, s3, v8
	v_add_u32_e32 v10, s3, v9
	s_lshl_b32 s25, s12, 2
	s_mul_i32 s4, s12, 0x74
	v_mad_u64_u32 v[2:3], s[18:19], v1, s12, v[2:3]
	v_add_u32_e32 v1, s25, v2
	v_add_u32_e32 v3, s25, v1
	v_add_u32_e32 v11, s25, v3
	v_add_u32_e32 v12, s4, v11
	v_add_u32_e32 v13, s25, v12
	v_add_u32_e32 v14, s25, v13
	s_add_i32 s12, s10, 63
	s_and_b32 s21, s15, 0xffff
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	s_mul_i32 s4, s24, 0x420
	s_add_i32 s10, s4, 0
	v_lshlrev_b32_e32 v4, 1, v4
	s_mov_b32 m0, s10
	s_nop 0
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	s_add_i32 s33, s4, 0x1080
	s_add_i32 m0, s10, 0x1080
	v_lshlrev_b32_e32 v5, 1, v5
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_i32 s34, s4, 0x2100
	s_add_i32 m0, s10, 0x2100
	v_lshlrev_b32_e32 v6, 1, v6
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	s_add_i32 s35, s4, 0x3180
	s_add_i32 m0, s10, 0x3180
	v_lshlrev_b32_e32 v7, 1, v7
	buffer_load_dwordx4 v7, s[20:23], 0 offen lds
	s_add_i32 s36, s4, 0x4200
	s_add_i32 m0, s10, 0x4200
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_load_dwordx4 v8, s[20:23], 0 offen lds
	s_add_i32 s37, s4, 0x5280
	s_add_i32 m0, s10, 0x5280
	v_lshlrev_b32_e32 v9, 1, v9
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	s_add_i32 s38, s4, 0x6300
	s_add_i32 m0, s10, 0x6300
	v_lshlrev_b32_e32 v15, 1, v10
	buffer_load_dwordx4 v15, s[20:23], 0 offen lds
	s_add_i32 s39, s4, 0x7380
	s_add_i32 m0, s10, 0x7380
	v_add_lshl_u32 v10, v10, s3, 1
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	s_and_b32 s17, s5, 0xffff
	s_mov_b32 s18, s22
	s_mov_b32 s19, s23
	s_add_i32 s3, 0, 0x107e0
	s_add_i32 s11, s3, s4
	v_lshlrev_b32_e32 v16, 1, v2
	s_mov_b32 m0, s11
	s_nop 0
	buffer_load_dwordx4 v16, s[16:19], 0 offen lds
	s_add_i32 m0, s3, s33
	v_lshlrev_b32_e32 v17, 1, v1
	buffer_load_dwordx4 v17, s[16:19], 0 offen lds
	s_add_i32 m0, s3, s34
	v_lshlrev_b32_e32 v18, 1, v3
	buffer_load_dwordx4 v18, s[16:19], 0 offen lds
	s_add_i32 m0, s3, s35
	v_lshlrev_b32_e32 v11, 1, v11
	buffer_load_dwordx4 v11, s[16:19], 0 offen lds
	s_add_i32 m0, s3, s36
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_load_dwordx4 v12, s[16:19], 0 offen lds
	s_add_i32 m0, s3, s37
	v_lshlrev_b32_e32 v13, 1, v13
	buffer_load_dwordx4 v13, s[16:19], 0 offen lds
	s_add_i32 m0, s3, s38
	v_lshlrev_b32_e32 v19, 1, v14
	buffer_load_dwordx4 v19, s[16:19], 0 offen lds
	s_add_i32 m0, s3, s39
	v_add_lshl_u32 v14, v14, s25, 1
	buffer_load_dwordx4 v14, s[16:19], 0 offen lds
	s_add_u32 s28, s20, 0x80
	s_addc_u32 s17, s15, 0
	s_add_u32 s24, s16, 0x80
	s_addc_u32 s18, s5, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_b32 s29, s17, 0xffff
	s_mov_b32 s30, s22
	s_mov_b32 s31, s23
	s_add_i32 m0, s10, 0x8400
	s_nop 0
	buffer_load_dwordx4 v4, s[28:31], 0 offen lds
	s_add_i32 m0, s10, 0x9480
	s_nop 0
	buffer_load_dwordx4 v5, s[28:31], 0 offen lds
	s_add_i32 m0, s10, 0xa500
	s_nop 0
	buffer_load_dwordx4 v6, s[28:31], 0 offen lds
	s_add_i32 m0, s10, 0xb580
	s_nop 0
	buffer_load_dwordx4 v7, s[28:31], 0 offen lds
	s_add_i32 m0, s10, 0xc600
	s_nop 0
	buffer_load_dwordx4 v8, s[28:31], 0 offen lds
	s_add_i32 m0, s10, 0xd680
	s_nop 0
	buffer_load_dwordx4 v9, s[28:31], 0 offen lds
	s_add_i32 m0, s10, 0xe700
	s_nop 0
	buffer_load_dwordx4 v15, s[28:31], 0 offen lds
	s_add_i32 m0, s10, 0xf780
	s_nop 0
	buffer_load_dwordx4 v10, s[28:31], 0 offen lds
	s_and_b32 s25, s18, 0xffff
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	s_add_i32 s17, 0, 0x18be0
	s_add_i32 m0, s17, s4
	s_nop 0
	buffer_load_dwordx4 v16, s[24:27], 0 offen lds
	s_add_i32 m0, s17, s33
	s_nop 0
	buffer_load_dwordx4 v17, s[24:27], 0 offen lds
	s_add_i32 m0, s17, s34
	s_nop 0
	buffer_load_dwordx4 v18, s[24:27], 0 offen lds
	s_add_i32 m0, s17, s35
	s_nop 0
	buffer_load_dwordx4 v11, s[24:27], 0 offen lds
	s_add_i32 m0, s17, s36
	s_nop 0
	buffer_load_dwordx4 v12, s[24:27], 0 offen lds
	s_add_i32 m0, s17, s37
	s_nop 0
	buffer_load_dwordx4 v13, s[24:27], 0 offen lds
	s_add_i32 m0, s17, s38
	s_nop 0
	buffer_load_dwordx4 v19, s[24:27], 0 offen lds
	s_add_i32 m0, s17, s39
	s_nop 0
	buffer_load_dwordx4 v14, s[24:27], 0 offen lds
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	v_and_b32_e32 v3, 15, v0
	v_lshlrev_b32_e32 v1, 10, v3
	s_movk_i32 s17, 0xb0
	v_and_or_b32 v2, v254, s17, v1
	v_accvgpr_write_b32 a209, v3
	v_lshlrev_b32_e32 v3, 5, v3
	v_add_u32_e32 v2, v2, v3
	v_add_u32_e32 v252, 0, v2
	ds_read_b128 a[40:43], v252
	ds_read_b128 a[44:47], v252 offset:64
	ds_read_b128 a[32:35], v252 offset:256
	ds_read_b128 a[36:39], v252 offset:320
	ds_read_b128 a[24:27], v252 offset:512
	ds_read_b128 a[28:31], v252 offset:576
	ds_read_b128 a[16:19], v252 offset:768
	ds_read_b128 a[20:23], v252 offset:832
	ds_read_b128 a[8:11], v252 offset:16896
	ds_read_b128 a[12:15], v252 offset:16960
	ds_read_b128 a[0:3], v252 offset:17152
	ds_read_b128 a[4:7], v252 offset:17216
	ds_read_b128 a[120:123], v252 offset:17408
	ds_read_b128 a[112:115], v252 offset:17472
	ds_read_b128 a[52:55], v252 offset:17664
	ds_read_b128 a[48:51], v252 offset:17728
	v_and_b32_e32 v0, 48, v0
	s_lshl_b32 s17, s9, 1
	s_and_b32 s17, s17, 0x80
	v_or3_b32 v0, v0, s17, v1
	v_add_u32_e32 v0, v0, v3
	v_add_u32_e32 v253, s3, v0
	ds_read_b128 a[124:127], v253
	ds_read_b128 a[116:119], v253 offset:64
	ds_read_b128 a[108:111], v253 offset:256
	ds_read_b128 a[104:107], v253 offset:320
	ds_read_b128 a[100:103], v253 offset:512
	ds_read_b128 a[96:99], v253 offset:576
	ds_read_b128 a[92:95], v253 offset:768
	ds_read_b128 a[88:91], v253 offset:832
	ds_read_b128 a[84:87], v253 offset:16896
	ds_read_b128 a[80:83], v253 offset:16960
	ds_read_b128 a[76:79], v253 offset:17152
	ds_read_b128 a[72:75], v253 offset:17216
	ds_read_b128 a[68:71], v253 offset:17408
	ds_read_b128 a[64:67], v253 offset:17472
	ds_read_b128 a[60:63], v253 offset:17664
	ds_read_b128 a[56:59], v253 offset:17728
	s_cmpk_lt_i32 s12, 0x80
	s_cbranch_scc1 .LBB0_5
; %bb.1:                                ; %.lr.ph
	v_accvgpr_write_b32 a227, v0
	v_accvgpr_write_b32 a226, v2
	v_accvgpr_write_b32 a225, v14
	v_accvgpr_write_b32 a224, v19
	v_accvgpr_write_b32 a223, v13
	v_accvgpr_write_b32 a222, v12
	v_accvgpr_write_b32 a221, v11
	v_accvgpr_write_b32 a220, v18
	v_accvgpr_write_b32 a219, v17
	v_accvgpr_write_b32 a218, v16
	v_accvgpr_write_b32 a217, v10
	v_accvgpr_write_b32 a216, v15
	v_accvgpr_write_b32 a215, v9
	v_accvgpr_write_b32 a214, v8
	s_lshr_b32 s12, s12, 6
	s_add_u32 s16, s16, 0x100
	s_addc_u32 s17, s5, 0
	s_add_u32 s20, s20, 0x100
	s_addc_u32 s21, s15, 0
	s_add_i32 s5, s12, -2
	s_cmp_eq_u32 s12, 2
	v_accvgpr_write_b32 a210, v4
	v_accvgpr_write_b32 a211, v5
	v_accvgpr_write_b32 a212, v6
	v_accvgpr_write_b32 a213, v7
	s_cbranch_scc1 .LBB0_6
; %bb.2:                                ; %.lr.ph.split
	v_accvgpr_write_b32 a208, v254
	s_mov_b32 s12, 0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a231, 0
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_accvgpr_mov_b32 a236, a214
	v_accvgpr_mov_b32 a237, a215
	v_accvgpr_mov_b32 a242, a216
	v_accvgpr_mov_b32 a247, a217
	v_accvgpr_mov_b32 a238, a218
	v_accvgpr_mov_b32 a243, a219
	v_accvgpr_mov_b32 a239, a220
	v_accvgpr_mov_b32 a244, a221
	v_accvgpr_mov_b32 a240, a222
	v_accvgpr_mov_b32 a245, a223
	v_accvgpr_mov_b32 a241, a224
	v_accvgpr_mov_b32 a246, a225
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s14, s12, 1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[0:3], a[124:127], a[40:43], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[116:119], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[248:251], a[108:111], a[40:43], v[248:251]
	v_mfma_f32_16x16x32_f16 v[248:251], a[104:107], a[44:47], v[248:251]
	v_mfma_f32_16x16x32_f16 v[244:247], a[100:103], a[40:43], v[244:247]
	v_mfma_f32_16x16x32_f16 v[244:247], a[96:99], a[44:47], v[244:247]
	v_mfma_f32_16x16x32_f16 v[240:243], a[92:95], a[40:43], v[240:243]
	v_mfma_f32_16x16x32_f16 v[240:243], a[88:91], a[44:47], v[240:243]
	v_mfma_f32_16x16x32_f16 v[236:239], a[84:87], a[40:43], v[236:239]
	v_mfma_f32_16x16x32_f16 v[236:239], a[80:83], a[44:47], v[236:239]
	v_mfma_f32_16x16x32_f16 v[232:235], a[76:79], a[40:43], v[232:235]
	v_mfma_f32_16x16x32_f16 v[232:235], a[72:75], a[44:47], v[232:235]
	v_mfma_f32_16x16x32_f16 v[228:231], a[68:71], a[40:43], v[228:231]
	v_mfma_f32_16x16x32_f16 v[228:231], a[64:67], a[44:47], v[228:231]
	v_mfma_f32_16x16x32_f16 v[224:227], a[60:63], a[40:43], v[224:227]
	v_mfma_f32_16x16x32_f16 v[224:227], a[56:59], a[44:47], v[224:227]
	v_mfma_f32_16x16x32_f16 v[220:223], a[124:127], a[32:35], v[220:223]
	v_mfma_f32_16x16x32_f16 v[220:223], a[116:119], a[36:39], v[220:223]
	v_mfma_f32_16x16x32_f16 v[216:219], a[108:111], a[32:35], v[216:219]
	v_mfma_f32_16x16x32_f16 v[216:219], a[104:107], a[36:39], v[216:219]
	v_mfma_f32_16x16x32_f16 v[212:215], a[100:103], a[32:35], v[212:215]
	v_mfma_f32_16x16x32_f16 v[212:215], a[96:99], a[36:39], v[212:215]
	v_mfma_f32_16x16x32_f16 v[208:211], a[92:95], a[32:35], v[208:211]
	v_mfma_f32_16x16x32_f16 v[208:211], a[88:91], a[36:39], v[208:211]
	v_mfma_f32_16x16x32_f16 v[204:207], a[84:87], a[32:35], v[204:207]
	v_mfma_f32_16x16x32_f16 v[204:207], a[80:83], a[36:39], v[204:207]
	v_mfma_f32_16x16x32_f16 v[200:203], a[76:79], a[32:35], v[200:203]
	v_mfma_f32_16x16x32_f16 v[200:203], a[72:75], a[36:39], v[200:203]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_and_b32 s25, s21, 0xffff
	s_mov_b32 s24, s20
	s_mul_i32 s15, s14, 0x8400
	s_add_i32 s18, s10, s15
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[196:199], a[68:71], a[32:35], v[196:199]
	v_mfma_f32_16x16x32_f16 v[196:199], a[64:67], a[36:39], v[196:199]
	v_mfma_f32_16x16x32_f16 v[192:195], a[60:63], a[32:35], v[192:195]
	v_mfma_f32_16x16x32_f16 v[192:195], a[56:59], a[36:39], v[192:195]
	s_add_i32 m0, s18, 0x1080
	s_nop 0
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[188:191], a[124:127], a[24:27], v[188:191]
	v_mfma_f32_16x16x32_f16 v[188:191], a[116:119], a[28:31], v[188:191]
	v_mfma_f32_16x16x32_f16 v[184:187], a[108:111], a[24:27], v[184:187]
	v_mfma_f32_16x16x32_f16 v[184:187], a[104:107], a[28:31], v[184:187]
	s_add_i32 m0, s18, 0x2100
	s_nop 0
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[180:183], a[100:103], a[24:27], v[180:183]
	v_mfma_f32_16x16x32_f16 v[180:183], a[96:99], a[28:31], v[180:183]
	v_mfma_f32_16x16x32_f16 v[176:179], a[92:95], a[24:27], v[176:179]
	v_mfma_f32_16x16x32_f16 v[176:179], a[88:91], a[28:31], v[176:179]
	s_add_i32 m0, s18, 0x3180
	s_nop 0
	buffer_load_dwordx4 v7, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[172:175], a[84:87], a[24:27], v[172:175]
	v_mfma_f32_16x16x32_f16 v[172:175], a[80:83], a[28:31], v[172:175]
	v_mfma_f32_16x16x32_f16 v[168:171], a[76:79], a[24:27], v[168:171]
	v_mfma_f32_16x16x32_f16 v[168:171], a[72:75], a[28:31], v[168:171]
	s_add_i32 m0, s18, 0x4200
	v_accvgpr_read_b32 v4, a236
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[164:167], a[68:71], a[24:27], v[164:167]
	v_mfma_f32_16x16x32_f16 v[164:167], a[64:67], a[28:31], v[164:167]
	v_mfma_f32_16x16x32_f16 v[160:163], a[60:63], a[24:27], v[160:163]
	v_mfma_f32_16x16x32_f16 v[160:163], a[56:59], a[28:31], v[160:163]
	s_add_i32 m0, s18, 0x5280
	v_accvgpr_read_b32 v4, a237
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[156:159], a[124:127], a[16:19], v[156:159]
	v_mfma_f32_16x16x32_f16 v[156:159], a[116:119], a[20:23], v[156:159]
	v_mfma_f32_16x16x32_f16 v[152:155], a[108:111], a[16:19], v[152:155]
	v_mfma_f32_16x16x32_f16 v[152:155], a[104:107], a[20:23], v[152:155]
	s_add_i32 m0, s18, 0x6300
	v_accvgpr_read_b32 v4, a242
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[148:151], a[100:103], a[16:19], v[148:151]
	v_mfma_f32_16x16x32_f16 v[148:151], a[96:99], a[20:23], v[148:151]
	v_mfma_f32_16x16x32_f16 v[144:147], a[92:95], a[16:19], v[144:147]
	v_mfma_f32_16x16x32_f16 v[144:147], a[88:91], a[20:23], v[144:147]
	s_add_i32 m0, s18, 0x7380
	v_accvgpr_read_b32 v4, a247
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[140:143], a[84:87], a[16:19], v[140:143]
	v_mfma_f32_16x16x32_f16 v[140:143], a[80:83], a[20:23], v[140:143]
	v_mfma_f32_16x16x32_f16 v[136:139], a[76:79], a[16:19], v[136:139]
	v_mfma_f32_16x16x32_f16 v[136:139], a[72:75], a[20:23], v[136:139]
	s_and_b32 s25, s17, 0xffff
	s_mov_b32 s24, s16
	s_add_i32 s15, s11, s15
	s_mov_b32 m0, s15
	v_accvgpr_read_b32 v4, a238
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[132:135], a[68:71], a[16:19], v[132:135]
	v_mfma_f32_16x16x32_f16 v[132:135], a[64:67], a[20:23], v[132:135]
	v_mfma_f32_16x16x32_f16 v[128:131], a[60:63], a[16:19], v[128:131]
	v_mfma_f32_16x16x32_f16 v[128:131], a[56:59], a[20:23], v[128:131]
	s_add_i32 m0, s15, 0x1080
	v_accvgpr_read_b32 v4, a243
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[124:127], a[124:127], a[8:11], v[124:127]
	v_mfma_f32_16x16x32_f16 v[124:127], a[116:119], a[12:15], v[124:127]
	v_mfma_f32_16x16x32_f16 v[120:123], a[108:111], a[8:11], v[120:123]
	v_mfma_f32_16x16x32_f16 v[120:123], a[104:107], a[12:15], v[120:123]
	s_add_i32 m0, s15, 0x2100
	v_accvgpr_read_b32 v4, a239
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[116:119], a[100:103], a[8:11], v[116:119]
	v_mfma_f32_16x16x32_f16 v[116:119], a[96:99], a[12:15], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[92:95], a[8:11], v[112:115]
	v_mfma_f32_16x16x32_f16 v[112:115], a[88:91], a[12:15], v[112:115]
	s_add_i32 m0, s15, 0x3180
	v_accvgpr_read_b32 v4, a244
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[108:111], a[84:87], a[8:11], v[108:111]
	v_mfma_f32_16x16x32_f16 v[108:111], a[80:83], a[12:15], v[108:111]
	v_mfma_f32_16x16x32_f16 v[104:107], a[76:79], a[8:11], v[104:107]
	v_mfma_f32_16x16x32_f16 v[104:107], a[72:75], a[12:15], v[104:107]
	s_add_i32 m0, s15, 0x4200
	v_accvgpr_read_b32 v4, a240
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[100:103], a[68:71], a[8:11], v[100:103]
	v_mfma_f32_16x16x32_f16 v[100:103], a[64:67], a[12:15], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[60:63], a[8:11], v[96:99]
	v_mfma_f32_16x16x32_f16 v[96:99], a[56:59], a[12:15], v[96:99]
	s_add_i32 m0, s15, 0x5280
	v_accvgpr_read_b32 v4, a245
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[92:95], a[124:127], a[0:3], v[92:95]
	v_mfma_f32_16x16x32_f16 v[92:95], a[116:119], a[4:7], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[108:111], a[0:3], v[88:91]
	v_mfma_f32_16x16x32_f16 v[88:91], a[104:107], a[4:7], v[88:91]
	s_add_i32 m0, s15, 0x6300
	v_accvgpr_read_b32 v4, a241
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[84:87], a[100:103], a[0:3], v[84:87]
	v_mfma_f32_16x16x32_f16 v[84:87], a[96:99], a[4:7], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[92:95], a[0:3], v[80:83]
	v_mfma_f32_16x16x32_f16 v[80:83], a[88:91], a[4:7], v[80:83]
	s_add_i32 m0, s15, 0x7380
	v_accvgpr_read_b32 v4, a246
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_mfma_f32_16x16x32_f16 v[76:79], a[84:87], a[0:3], v[76:79]
	v_mfma_f32_16x16x32_f16 v[76:79], a[80:83], a[4:7], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[76:79], a[0:3], v[72:75]
	v_mfma_f32_16x16x32_f16 v[72:75], a[72:75], a[4:7], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[68:71], a[0:3], v[68:71]
	v_mfma_f32_16x16x32_f16 v[68:71], a[64:67], a[4:7], v[68:71]
	s_lshl_b32 s14, s14, 14
	s_xor_b32 s14, s14, 0x4000
	s_lshr_b32 s15, s14, 5
	s_or_b32 s14, s15, s14
	s_lshl_b32 s14, s14, 1
	v_add_u32_e32 v4, s14, v252
	ds_read_b128 a[40:43], v4
	v_mfma_f32_16x16x32_f16 v[64:67], a[60:63], a[0:3], v[64:67]
	ds_read_b128 a[44:47], v4 offset:64
	v_mfma_f32_16x16x32_f16 v[64:67], a[56:59], a[4:7], v[64:67]
	ds_read_b128 a[32:35], v4 offset:256
	v_mfma_f32_16x16x32_f16 v[60:63], a[124:127], a[120:123], v[60:63]
	ds_read_b128 a[36:39], v4 offset:320
	v_mfma_f32_16x16x32_f16 v[60:63], a[116:119], a[112:115], v[60:63]
	ds_read_b128 a[24:27], v4 offset:512
	v_mfma_f32_16x16x32_f16 v[56:59], a[108:111], a[120:123], v[56:59]
	ds_read_b128 a[28:31], v4 offset:576
	v_mfma_f32_16x16x32_f16 v[56:59], a[104:107], a[112:115], v[56:59]
	ds_read_b128 a[16:19], v4 offset:768
	v_mfma_f32_16x16x32_f16 v[52:55], a[100:103], a[120:123], v[52:55]
	ds_read_b128 a[20:23], v4 offset:832
	v_mfma_f32_16x16x32_f16 v[52:55], a[96:99], a[112:115], v[52:55]
	ds_read_b128 a[8:11], v4 offset:16896
	v_mfma_f32_16x16x32_f16 v[48:51], a[92:95], a[120:123], v[48:51]
	ds_read_b128 a[12:15], v4 offset:16960
	v_mfma_f32_16x16x32_f16 v[48:51], a[88:91], a[112:115], v[48:51]
	ds_read_b128 a[0:3], v4 offset:17152
	v_mfma_f32_16x16x32_f16 v[44:47], a[84:87], a[120:123], v[44:47]
	ds_read_b128 a[4:7], v4 offset:17216
	v_mfma_f32_16x16x32_f16 v[44:47], a[80:83], a[112:115], v[44:47]
	ds_read_b128 a[128:131], v4 offset:17408
	v_mfma_f32_16x16x32_f16 v[40:43], a[76:79], a[120:123], v[40:43]
	ds_read_b128 a[132:135], v4 offset:17472
	v_mfma_f32_16x16x32_f16 v[40:43], a[72:75], a[112:115], v[40:43]
	ds_read_b128 a[136:139], v4 offset:17664
	v_mfma_f32_16x16x32_f16 v[36:39], a[68:71], a[120:123], v[36:39]
	ds_read_b128 a[140:143], v4 offset:17728
	v_mfma_f32_16x16x32_f16 v[36:39], a[64:67], a[112:115], v[36:39]
	v_add_u32_e32 v254, s14, v253
	ds_read_b128 a[144:147], v254
	v_mfma_f32_16x16x32_f16 v[32:35], a[60:63], a[120:123], v[32:35]
	ds_read_b128 a[148:151], v254 offset:64
	v_mfma_f32_16x16x32_f16 v[32:35], a[56:59], a[112:115], v[32:35]
	ds_read_b128 a[152:155], v254 offset:256
	v_mfma_f32_16x16x32_f16 v[28:31], a[124:127], a[52:55], v[28:31]
	ds_read_b128 a[156:159], v254 offset:320
	v_mfma_f32_16x16x32_f16 v[28:31], a[116:119], a[48:51], v[28:31]
	ds_read_b128 a[160:163], v254 offset:512
	v_mfma_f32_16x16x32_f16 v[24:27], a[108:111], a[52:55], v[24:27]
	ds_read_b128 a[164:167], v254 offset:576
	v_mfma_f32_16x16x32_f16 v[24:27], a[104:107], a[48:51], v[24:27]
	ds_read_b128 a[168:171], v254 offset:768
	v_mfma_f32_16x16x32_f16 v[20:23], a[100:103], a[52:55], v[20:23]
	ds_read_b128 a[172:175], v254 offset:832
	v_mfma_f32_16x16x32_f16 v[20:23], a[96:99], a[48:51], v[20:23]
	ds_read_b128 a[176:179], v254 offset:16896
	v_mfma_f32_16x16x32_f16 v[16:19], a[92:95], a[52:55], v[16:19]
	ds_read_b128 a[180:183], v254 offset:16960
	v_mfma_f32_16x16x32_f16 v[16:19], a[88:91], a[48:51], v[16:19]
	ds_read_b128 a[184:187], v254 offset:17152
	v_mfma_f32_16x16x32_f16 a[84:87], a[84:87], a[52:55], a[252:255]
	ds_read_b128 a[188:191], v254 offset:17216
	v_mfma_f32_16x16x32_f16 a[252:255], a[80:83], a[48:51], a[84:87]
	ds_read_b128 a[192:195], v254 offset:17408
	v_mfma_f32_16x16x32_f16 a[76:79], a[76:79], a[52:55], a[248:251]
	ds_read_b128 a[196:199], v254 offset:17472
	v_mfma_f32_16x16x32_f16 a[248:251], a[72:75], a[48:51], a[76:79]
	ds_read_b128 a[200:203], v254 offset:17664
	v_mfma_f32_16x16x32_f16 a[232:235], a[68:71], a[52:55], a[232:235]
	ds_read_b128 a[204:207], v254 offset:17728
	v_mfma_f32_16x16x32_f16 a[232:235], a[64:67], a[48:51], a[232:235]
	v_mfma_f32_16x16x32_f16 a[228:231], a[60:63], a[52:55], a[228:231]
	v_mfma_f32_16x16x32_f16 a[228:231], a[56:59], a[48:51], a[228:231]
	v_accvgpr_read_b32 v7, a213
	v_accvgpr_read_b32 v6, a212
	v_accvgpr_read_b32 v5, a211
	v_accvgpr_read_b32 v4, a210
	s_add_u32 s20, s20, 0x80
	s_addc_u32 s21, s21, 0
	s_add_u32 s16, s16, 0x80
	s_addc_u32 s17, s17, 0
	s_add_i32 s12, s12, 1
	s_cmp_lg_u32 s5, s12
	s_waitcnt lgkmcnt(14)
	v_accvgpr_mov_b32 a120, a128
	v_accvgpr_mov_b32 a121, a129
	v_accvgpr_mov_b32 a122, a130
	v_accvgpr_mov_b32 a123, a131
	v_accvgpr_mov_b32 a112, a132
	v_accvgpr_mov_b32 a113, a133
	v_accvgpr_mov_b32 a114, a134
	v_accvgpr_mov_b32 a115, a135
	v_accvgpr_mov_b32 a52, a136
	v_accvgpr_mov_b32 a53, a137
	v_accvgpr_mov_b32 a54, a138
	v_accvgpr_mov_b32 a55, a139
	v_accvgpr_mov_b32 a48, a140
	v_accvgpr_mov_b32 a49, a141
	v_accvgpr_mov_b32 a50, a142
	v_accvgpr_mov_b32 a51, a143
	v_accvgpr_mov_b32 a124, a144
	v_accvgpr_mov_b32 a125, a145
	v_accvgpr_mov_b32 a126, a146
	v_accvgpr_mov_b32 a127, a147
	v_accvgpr_mov_b32 a116, a148
	v_accvgpr_mov_b32 a117, a149
	v_accvgpr_mov_b32 a118, a150
	v_accvgpr_mov_b32 a119, a151
	s_waitcnt lgkmcnt(13)
	v_accvgpr_mov_b32 a108, a152
	v_accvgpr_mov_b32 a109, a153
	v_accvgpr_mov_b32 a110, a154
	v_accvgpr_mov_b32 a111, a155
	s_waitcnt lgkmcnt(12)
	v_accvgpr_mov_b32 a104, a156
	v_accvgpr_mov_b32 a105, a157
	v_accvgpr_mov_b32 a106, a158
	v_accvgpr_mov_b32 a107, a159
	s_waitcnt lgkmcnt(11)
	v_accvgpr_mov_b32 a100, a160
	v_accvgpr_mov_b32 a101, a161
	v_accvgpr_mov_b32 a102, a162
	v_accvgpr_mov_b32 a103, a163
	s_waitcnt lgkmcnt(10)
	v_accvgpr_mov_b32 a96, a164
	v_accvgpr_mov_b32 a97, a165
	v_accvgpr_mov_b32 a98, a166
	v_accvgpr_mov_b32 a99, a167
	s_waitcnt lgkmcnt(9)
	v_accvgpr_mov_b32 a92, a168
	v_accvgpr_mov_b32 a93, a169
	v_accvgpr_mov_b32 a94, a170
	v_accvgpr_mov_b32 a95, a171
	s_waitcnt lgkmcnt(8)
	v_accvgpr_mov_b32 a88, a172
	v_accvgpr_mov_b32 a89, a173
	v_accvgpr_mov_b32 a90, a174
	v_accvgpr_mov_b32 a91, a175
	s_waitcnt lgkmcnt(7)
	v_accvgpr_mov_b32 a84, a176
	v_accvgpr_mov_b32 a85, a177
	v_accvgpr_mov_b32 a86, a178
	v_accvgpr_mov_b32 a87, a179
	s_waitcnt lgkmcnt(6)
	v_accvgpr_mov_b32 a80, a180
	v_accvgpr_mov_b32 a81, a181
	v_accvgpr_mov_b32 a82, a182
	v_accvgpr_mov_b32 a83, a183
	s_waitcnt lgkmcnt(5)
	v_accvgpr_mov_b32 a76, a184
	v_accvgpr_mov_b32 a77, a185
	v_accvgpr_mov_b32 a78, a186
	v_accvgpr_mov_b32 a79, a187
	s_waitcnt lgkmcnt(4)
	v_accvgpr_mov_b32 a72, a188
	v_accvgpr_mov_b32 a73, a189
	v_accvgpr_mov_b32 a74, a190
	v_accvgpr_mov_b32 a75, a191
	s_waitcnt lgkmcnt(3)
	v_accvgpr_mov_b32 a68, a192
	v_accvgpr_mov_b32 a69, a193
	v_accvgpr_mov_b32 a70, a194
	v_accvgpr_mov_b32 a71, a195
	s_waitcnt lgkmcnt(2)
	v_accvgpr_mov_b32 a64, a196
	v_accvgpr_mov_b32 a65, a197
	v_accvgpr_mov_b32 a66, a198
	v_accvgpr_mov_b32 a67, a199
	s_waitcnt lgkmcnt(1)
	v_accvgpr_mov_b32 a60, a200
	v_accvgpr_mov_b32 a61, a201
	v_accvgpr_mov_b32 a62, a202
	v_accvgpr_mov_b32 a63, a203
	s_waitcnt lgkmcnt(0)
	v_accvgpr_mov_b32 a56, a204
	v_accvgpr_mov_b32 a57, a205
	v_accvgpr_mov_b32 a58, a206
	v_accvgpr_mov_b32 a59, a207
	s_cbranch_scc1 .LBB0_3
; %bb.4:                                ; %Flow
	v_accvgpr_mov_b32 a59, a207
	v_accvgpr_mov_b32 a58, a206
	v_accvgpr_mov_b32 a57, a205
	v_accvgpr_mov_b32 a56, a204
	v_accvgpr_mov_b32 a63, a203
	v_accvgpr_mov_b32 a62, a202
	v_accvgpr_mov_b32 a61, a201
	v_accvgpr_mov_b32 a60, a200
	v_accvgpr_mov_b32 a67, a199
	v_accvgpr_mov_b32 a66, a198
	v_accvgpr_mov_b32 a65, a197
	v_accvgpr_mov_b32 a64, a196
	v_accvgpr_mov_b32 a71, a195
	v_accvgpr_mov_b32 a70, a194
	v_accvgpr_mov_b32 a69, a193
	v_accvgpr_mov_b32 a68, a192
	v_accvgpr_mov_b32 a75, a191
	v_accvgpr_mov_b32 a74, a190
	v_accvgpr_mov_b32 a73, a189
	v_accvgpr_mov_b32 a72, a188
	v_accvgpr_mov_b32 a79, a187
	v_accvgpr_mov_b32 a78, a186
	v_accvgpr_mov_b32 a77, a185
	v_accvgpr_mov_b32 a76, a184
	v_accvgpr_mov_b32 a83, a183
	v_accvgpr_mov_b32 a82, a182
	v_accvgpr_mov_b32 a81, a181
	v_accvgpr_mov_b32 a80, a180
	v_accvgpr_mov_b32 a87, a179
	v_accvgpr_mov_b32 a86, a178
	v_accvgpr_mov_b32 a85, a177
	v_accvgpr_mov_b32 a84, a176
	v_accvgpr_mov_b32 a91, a175
	v_accvgpr_mov_b32 a90, a174
	v_accvgpr_mov_b32 a89, a173
	v_accvgpr_mov_b32 a88, a172
	v_accvgpr_mov_b32 a95, a171
	v_accvgpr_mov_b32 a94, a170
	v_accvgpr_mov_b32 a93, a169
	v_accvgpr_mov_b32 a92, a168
	v_accvgpr_mov_b32 a99, a167
	v_accvgpr_mov_b32 a98, a166
	v_accvgpr_mov_b32 a97, a165
	v_accvgpr_mov_b32 a96, a164
	v_accvgpr_mov_b32 a103, a163
	v_accvgpr_mov_b32 a102, a162
	v_accvgpr_mov_b32 a101, a161
	v_accvgpr_mov_b32 a100, a160
	v_accvgpr_mov_b32 a107, a159
	v_accvgpr_mov_b32 a106, a158
	v_accvgpr_mov_b32 a105, a157
	v_accvgpr_mov_b32 a104, a156
	v_accvgpr_mov_b32 a111, a155
	v_accvgpr_mov_b32 a110, a154
	v_accvgpr_mov_b32 a109, a153
	v_accvgpr_mov_b32 a108, a152
	v_accvgpr_mov_b32 a119, a151
	v_accvgpr_mov_b32 a118, a150
	v_accvgpr_mov_b32 a117, a149
	v_accvgpr_mov_b32 a116, a148
	v_accvgpr_mov_b32 a127, a147
	v_accvgpr_mov_b32 a126, a146
	v_accvgpr_mov_b32 a125, a145
	v_accvgpr_mov_b32 a124, a144
	v_accvgpr_mov_b32 a51, a143
	v_accvgpr_mov_b32 a50, a142
	v_accvgpr_mov_b32 a49, a141
	v_accvgpr_mov_b32 a48, a140
	v_accvgpr_mov_b32 a55, a139
	v_accvgpr_mov_b32 a54, a138
	v_accvgpr_mov_b32 a53, a137
	v_accvgpr_mov_b32 a52, a136
	v_accvgpr_mov_b32 a115, a135
	v_accvgpr_mov_b32 a114, a134
	v_accvgpr_mov_b32 a113, a133
	v_accvgpr_mov_b32 a112, a132
	v_accvgpr_mov_b32 a123, a131
	v_accvgpr_mov_b32 a122, a130
	v_accvgpr_mov_b32 a121, a129
	v_accvgpr_mov_b32 a120, a128
	s_mov_b32 s14, s5
	v_accvgpr_read_b32 v254, a208
	s_branch .LBB0_7
.LBB0_5:
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a184, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v0, 0
	s_branch .LBB0_8
.LBB0_6:
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a252, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v0, 0
.LBB0_7:                                ; %._crit_edge.loopexit.peel.begin
	s_and_b32 s10, s14, 1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[0:3], a[124:127], a[40:43], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[116:119], a[44:47], v[0:3]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[40:43], v[248:251]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x32_f16 v[248:251], a[104:107], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[40:43], v[244:247]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_f16 v[244:247], a[96:99], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[40:43], v[240:243]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_f16 v[240:243], a[88:91], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[40:43], v[236:239]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[236:239], a[80:83], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[40:43], v[232:235]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[232:235], a[72:75], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[40:43], v[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[228:231], a[64:67], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[40:43], v[224:227]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[224:227], a[56:59], a[44:47], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[32:35], v[220:223]
	v_mfma_f32_16x16x32_f16 v[220:223], a[116:119], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[32:35], v[216:219]
	v_mfma_f32_16x16x32_f16 v[216:219], a[104:107], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[32:35], v[212:215]
	v_mfma_f32_16x16x32_f16 v[212:215], a[96:99], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[32:35], v[208:211]
	v_mfma_f32_16x16x32_f16 v[208:211], a[88:91], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[32:35], v[204:207]
	v_mfma_f32_16x16x32_f16 v[204:207], a[80:83], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[32:35], v[200:203]
	v_mfma_f32_16x16x32_f16 v[200:203], a[72:75], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[32:35], v[196:199]
	v_mfma_f32_16x16x32_f16 v[196:199], a[64:67], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[32:35], v[192:195]
	v_mfma_f32_16x16x32_f16 v[192:195], a[56:59], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[24:27], v[188:191]
	v_mfma_f32_16x16x32_f16 v[188:191], a[116:119], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[24:27], v[184:187]
	v_mfma_f32_16x16x32_f16 v[184:187], a[104:107], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[24:27], v[180:183]
	v_mfma_f32_16x16x32_f16 v[180:183], a[96:99], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[24:27], v[176:179]
	v_mfma_f32_16x16x32_f16 v[176:179], a[88:91], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[24:27], v[172:175]
	v_mfma_f32_16x16x32_f16 v[172:175], a[80:83], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[24:27], v[168:171]
	v_mfma_f32_16x16x32_f16 v[168:171], a[72:75], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[24:27], v[164:167]
	v_mfma_f32_16x16x32_f16 v[164:167], a[64:67], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[24:27], v[160:163]
	v_mfma_f32_16x16x32_f16 v[160:163], a[56:59], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[16:19], v[156:159]
	v_mfma_f32_16x16x32_f16 v[156:159], a[116:119], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[16:19], v[152:155]
	v_mfma_f32_16x16x32_f16 v[152:155], a[104:107], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[16:19], v[148:151]
	v_mfma_f32_16x16x32_f16 v[148:151], a[96:99], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[16:19], v[144:147]
	v_mfma_f32_16x16x32_f16 v[144:147], a[88:91], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[16:19], v[140:143]
	v_mfma_f32_16x16x32_f16 v[140:143], a[80:83], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[16:19], v[136:139]
	v_mfma_f32_16x16x32_f16 v[136:139], a[72:75], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[16:19], v[132:135]
	v_mfma_f32_16x16x32_f16 v[132:135], a[64:67], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[16:19], v[128:131]
	v_mfma_f32_16x16x32_f16 v[128:131], a[56:59], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[8:11], v[124:127]
	v_mfma_f32_16x16x32_f16 v[124:127], a[116:119], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[8:11], v[120:123]
	v_mfma_f32_16x16x32_f16 v[120:123], a[104:107], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[8:11], v[116:119]
	v_mfma_f32_16x16x32_f16 v[116:119], a[96:99], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[8:11], v[112:115]
	v_mfma_f32_16x16x32_f16 v[112:115], a[88:91], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[8:11], v[108:111]
	v_mfma_f32_16x16x32_f16 v[108:111], a[80:83], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[8:11], v[104:107]
	v_mfma_f32_16x16x32_f16 v[104:107], a[72:75], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[8:11], v[100:103]
	v_mfma_f32_16x16x32_f16 v[100:103], a[64:67], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[8:11], v[96:99]
	v_mfma_f32_16x16x32_f16 v[96:99], a[56:59], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[0:3], v[92:95]
	v_mfma_f32_16x16x32_f16 v[92:95], a[116:119], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[0:3], v[88:91]
	v_mfma_f32_16x16x32_f16 v[88:91], a[104:107], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[0:3], v[84:87]
	v_mfma_f32_16x16x32_f16 v[84:87], a[96:99], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[0:3], v[80:83]
	v_mfma_f32_16x16x32_f16 v[80:83], a[88:91], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[0:3], v[76:79]
	v_mfma_f32_16x16x32_f16 v[76:79], a[80:83], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[0:3], v[72:75]
	v_mfma_f32_16x16x32_f16 v[72:75], a[72:75], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[0:3], v[68:71]
	v_mfma_f32_16x16x32_f16 v[68:71], a[64:67], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[0:3], v[64:67]
	v_mfma_f32_16x16x32_f16 v[64:67], a[56:59], a[4:7], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[120:123], v[60:63]
	v_mfma_f32_16x16x32_f16 v[4:7], a[116:119], a[112:115], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a187, v7
	v_accvgpr_write_b32 a186, v6
	v_accvgpr_write_b32 a185, v5
	v_accvgpr_write_b32 a184, v4
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[120:123], v[56:59]
	v_mfma_f32_16x16x32_f16 v[4:7], a[104:107], a[112:115], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a183, v7
	v_accvgpr_write_b32 a182, v6
	v_accvgpr_write_b32 a181, v5
	v_accvgpr_write_b32 a180, v4
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[120:123], v[52:55]
	v_mfma_f32_16x16x32_f16 v[4:7], a[96:99], a[112:115], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a179, v7
	v_accvgpr_write_b32 a178, v6
	v_accvgpr_write_b32 a177, v5
	v_accvgpr_write_b32 a176, v4
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[120:123], v[48:51]
	v_mfma_f32_16x16x32_f16 v[4:7], a[88:91], a[112:115], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a175, v7
	v_accvgpr_write_b32 a174, v6
	v_accvgpr_write_b32 a173, v5
	v_accvgpr_write_b32 a172, v4
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[120:123], v[44:47]
	v_mfma_f32_16x16x32_f16 v[4:7], a[80:83], a[112:115], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a171, v7
	v_accvgpr_write_b32 a170, v6
	v_accvgpr_write_b32 a169, v5
	v_accvgpr_write_b32 a168, v4
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[120:123], v[40:43]
	v_mfma_f32_16x16x32_f16 v[4:7], a[72:75], a[112:115], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a167, v7
	v_accvgpr_write_b32 a166, v6
	v_accvgpr_write_b32 a165, v5
	v_accvgpr_write_b32 a164, v4
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[120:123], v[36:39]
	v_mfma_f32_16x16x32_f16 v[4:7], a[64:67], a[112:115], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a163, v7
	v_accvgpr_write_b32 a162, v6
	v_accvgpr_write_b32 a161, v5
	v_accvgpr_write_b32 a160, v4
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[120:123], v[32:35]
	v_mfma_f32_16x16x32_f16 v[4:7], a[56:59], a[112:115], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a159, v7
	v_accvgpr_write_b32 a158, v6
	v_accvgpr_write_b32 a157, v5
	v_accvgpr_write_b32 a156, v4
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[52:55], v[28:31]
	v_mfma_f32_16x16x32_f16 v[4:7], a[116:119], a[48:51], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a155, v7
	v_accvgpr_write_b32 a154, v6
	v_accvgpr_write_b32 a153, v5
	v_accvgpr_write_b32 a152, v4
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[52:55], v[24:27]
	v_mfma_f32_16x16x32_f16 v[4:7], a[104:107], a[48:51], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a151, v7
	v_accvgpr_write_b32 a150, v6
	v_accvgpr_write_b32 a149, v5
	v_accvgpr_write_b32 a148, v4
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[52:55], v[20:23]
	v_mfma_f32_16x16x32_f16 v[4:7], a[96:99], a[48:51], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a147, v7
	v_accvgpr_write_b32 a146, v6
	v_accvgpr_write_b32 a145, v5
	v_accvgpr_write_b32 a144, v4
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[52:55], v[16:19]
	v_mfma_f32_16x16x32_f16 v[4:7], a[88:91], a[48:51], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a143, v7
	v_accvgpr_write_b32 a142, v6
	v_accvgpr_write_b32 a141, v5
	v_accvgpr_write_b32 a140, v4
	v_mfma_f32_16x16x32_f16 a[136:139], a[84:87], a[52:55], a[252:255]
	v_mfma_f32_16x16x32_f16 a[136:139], a[80:83], a[48:51], a[136:139]
	v_mfma_f32_16x16x32_f16 a[132:135], a[76:79], a[52:55], a[248:251]
	v_mfma_f32_16x16x32_f16 a[132:135], a[72:75], a[48:51], a[132:135]
	v_accvgpr_mov_b32 a0, a232
	v_accvgpr_mov_b32 a1, a233
	v_accvgpr_mov_b32 a2, a234
	v_accvgpr_mov_b32 a3, a235
	s_nop 1
	v_mfma_f32_16x16x32_f16 a[0:3], a[68:71], a[52:55], a[0:3]
	v_mfma_f32_16x16x32_f16 a[128:131], a[64:67], a[48:51], a[0:3]
	v_accvgpr_read_b32 v4, a228
	v_accvgpr_read_b32 v5, a229
	v_accvgpr_read_b32 v6, a230
	v_accvgpr_read_b32 v7, a231
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[52:55], v[4:7]
	v_mfma_f32_16x16x32_f16 v[44:47], a[56:59], a[48:51], v[4:7]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_cmp_eq_u32 s14, s5
	s_cselect_b64 vcc, -1, 0
	s_mul_i32 s5, s10, 0x8400
	s_add_i32 s11, s5, 0
	s_and_b32 s21, s21, 0xffff
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	s_add_i32 s11, s11, s4
	v_bfrev_b32_e32 v4, 1
	v_accvgpr_read_b32 v5, a210
	v_cndmask_b32_e32 v5, v5, v4, vcc
	s_mov_b32 m0, s11
	s_nop 0
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_i32 m0, s11, 0x1080
	v_accvgpr_read_b32 v5, a211
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_i32 m0, s11, 0x2100
	v_accvgpr_read_b32 v5, a212
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_i32 m0, s11, 0x3180
	v_accvgpr_read_b32 v5, a213
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_i32 m0, s11, 0x4200
	v_accvgpr_read_b32 v5, a214
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_i32 m0, s11, 0x5280
	v_accvgpr_read_b32 v5, a215
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_i32 m0, s11, 0x6300
	v_accvgpr_read_b32 v5, a216
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_i32 m0, s11, 0x7380
	v_accvgpr_read_b32 v5, a217
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[20:23], 0 offen lds
	s_add_i32 s5, s3, s5
	s_and_b32 s17, s17, 0xffff
	s_mov_b32 s18, s22
	s_mov_b32 s19, s23
	s_add_i32 s4, s5, s4
	v_accvgpr_read_b32 v5, a218
	v_cndmask_b32_e32 v5, v5, v4, vcc
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_add_i32 m0, s4, 0x1080
	v_accvgpr_read_b32 v5, a219
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_add_i32 m0, s4, 0x2100
	v_accvgpr_read_b32 v5, a220
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_add_i32 m0, s4, 0x3180
	v_accvgpr_read_b32 v5, a221
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_add_i32 m0, s4, 0x4200
	v_accvgpr_read_b32 v5, a222
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_add_i32 m0, s4, 0x5280
	v_accvgpr_read_b32 v5, a223
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_add_i32 m0, s4, 0x6300
	v_accvgpr_read_b32 v5, a224
	v_cndmask_b32_e32 v5, v5, v4, vcc
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_add_i32 m0, s4, 0x7380
	v_accvgpr_read_b32 v5, a225
	v_cndmask_b32_e32 v4, v5, v4, vcc
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_lshl_b32 s4, s10, 14
	s_xor_b32 s4, s4, 0x4000
	s_lshr_b32 s5, s4, 5
	s_or_b32 s4, s5, s4
	s_lshl_b32 s4, s4, 1
	s_add_i32 s5, s4, 0
	v_accvgpr_read_b32 v4, a226
	v_add_u32_e32 v4, s5, v4
	ds_read_b128 a[40:43], v4
	ds_read_b128 a[44:47], v4 offset:64
	ds_read_b128 a[32:35], v4 offset:256
	ds_read_b128 a[36:39], v4 offset:320
	ds_read_b128 a[24:27], v4 offset:512
	ds_read_b128 a[28:31], v4 offset:576
	ds_read_b128 a[16:19], v4 offset:768
	ds_read_b128 a[20:23], v4 offset:832
	ds_read_b128 a[8:11], v4 offset:16896
	ds_read_b128 a[12:15], v4 offset:16960
	ds_read_b128 a[0:3], v4 offset:17152
	ds_read_b128 a[4:7], v4 offset:17216
	ds_read_b128 a[120:123], v4 offset:17408
	ds_read_b128 a[112:115], v4 offset:17472
	ds_read_b128 a[52:55], v4 offset:17664
	ds_read_b128 a[48:51], v4 offset:17728
	s_add_i32 s3, s3, s4
	v_accvgpr_read_b32 v4, a227
	v_add_u32_e32 v4, s3, v4
	ds_read_b128 a[124:127], v4
	ds_read_b128 a[116:119], v4 offset:64
	ds_read_b128 a[108:111], v4 offset:256
	ds_read_b128 a[104:107], v4 offset:320
	ds_read_b128 a[100:103], v4 offset:512
	ds_read_b128 a[96:99], v4 offset:576
	ds_read_b128 a[92:95], v4 offset:768
	ds_read_b128 a[88:91], v4 offset:832
	ds_read_b128 a[84:87], v4 offset:16896
	ds_read_b128 a[80:83], v4 offset:16960
	ds_read_b128 a[76:79], v4 offset:17152
	ds_read_b128 a[72:75], v4 offset:17216
	ds_read_b128 a[68:71], v4 offset:17408
	ds_read_b128 a[64:67], v4 offset:17472
	ds_read_b128 a[60:63], v4 offset:17664
	ds_read_b128 a[56:59], v4 offset:17728
.LBB0_8:                                ; %Flow394
	s_lshr_b32 s3, s9, 6
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[0:3], a[124:127], a[40:43], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[116:119], a[44:47], v[0:3]
	s_nop 7
	v_accvgpr_write_b32 a191, v3
	v_accvgpr_write_b32 a190, v2
	v_accvgpr_write_b32 a189, v1
	v_accvgpr_write_b32 a188, v0
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[40:43], v[248:251]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x32_f16 v[248:251], a[104:107], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[40:43], v[244:247]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_f16 v[244:247], a[96:99], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[40:43], v[240:243]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_f16 v[240:243], a[88:91], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[40:43], v[236:239]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[236:239], a[80:83], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[40:43], v[232:235]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[232:235], a[72:75], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[40:43], v[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[228:231], a[64:67], a[44:47], v[4:7]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[40:43], v[224:227]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[224:227], a[56:59], a[44:47], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[32:35], v[220:223]
	v_mfma_f32_16x16x32_f16 v[220:223], a[116:119], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[32:35], v[216:219]
	v_mfma_f32_16x16x32_f16 v[216:219], a[104:107], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[32:35], v[212:215]
	v_mfma_f32_16x16x32_f16 v[212:215], a[96:99], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[32:35], v[208:211]
	v_mfma_f32_16x16x32_f16 v[208:211], a[88:91], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[32:35], v[204:207]
	v_mfma_f32_16x16x32_f16 v[204:207], a[80:83], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[32:35], v[200:203]
	v_mfma_f32_16x16x32_f16 v[200:203], a[72:75], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[32:35], v[196:199]
	v_mfma_f32_16x16x32_f16 v[196:199], a[64:67], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[32:35], v[192:195]
	v_mfma_f32_16x16x32_f16 v[192:195], a[56:59], a[36:39], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[24:27], v[188:191]
	v_mfma_f32_16x16x32_f16 v[188:191], a[116:119], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[24:27], v[184:187]
	v_mfma_f32_16x16x32_f16 v[184:187], a[104:107], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[24:27], v[180:183]
	v_mfma_f32_16x16x32_f16 v[180:183], a[96:99], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[24:27], v[176:179]
	v_mfma_f32_16x16x32_f16 v[176:179], a[88:91], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[24:27], v[172:175]
	v_mfma_f32_16x16x32_f16 v[172:175], a[80:83], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[24:27], v[168:171]
	v_mfma_f32_16x16x32_f16 v[168:171], a[72:75], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[24:27], v[164:167]
	v_mfma_f32_16x16x32_f16 v[164:167], a[64:67], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[24:27], v[160:163]
	v_mfma_f32_16x16x32_f16 v[160:163], a[56:59], a[28:31], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[16:19], v[156:159]
	v_mfma_f32_16x16x32_f16 v[0:3], a[116:119], a[20:23], v[4:7]
	s_nop 7
	v_accvgpr_write_b32 a27, v3
	v_accvgpr_write_b32 a26, v2
	v_accvgpr_write_b32 a25, v1
	v_accvgpr_write_b32 a24, v0
	v_mfma_f32_16x16x32_f16 v[4:7], a[108:111], a[16:19], v[152:155]
	v_mfma_f32_16x16x32_f16 v[156:159], a[104:107], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[16:19], v[148:151]
	v_mfma_f32_16x16x32_f16 v[152:155], a[96:99], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[16:19], v[144:147]
	v_mfma_f32_16x16x32_f16 v[148:151], a[88:91], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[16:19], v[140:143]
	v_mfma_f32_16x16x32_f16 v[144:147], a[80:83], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[16:19], v[136:139]
	v_mfma_f32_16x16x32_f16 v[140:143], a[72:75], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[68:71], a[16:19], v[132:135]
	v_mfma_f32_16x16x32_f16 v[136:139], a[64:67], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[60:63], a[16:19], v[128:131]
	v_mfma_f32_16x16x32_f16 v[132:135], a[56:59], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[124:127], a[8:11], v[124:127]
	v_mfma_f32_16x16x32_f16 v[128:131], a[116:119], a[12:15], v[4:7]
	v_mfma_f32_16x16x32_f16 v[120:123], a[108:111], a[8:11], v[120:123]
	v_mfma_f32_16x16x32_f16 v[8:11], a[104:107], a[12:15], v[120:123]
	v_mfma_f32_16x16x32_f16 v[116:119], a[100:103], a[8:11], v[116:119]
	v_mfma_f32_16x16x32_f16 v[12:15], a[96:99], a[12:15], v[116:119]
	v_mfma_f32_16x16x32_f16 v[112:115], a[92:95], a[8:11], v[112:115]
	v_mfma_f32_16x16x32_f16 v[16:19], a[88:91], a[12:15], v[112:115]
	v_mfma_f32_16x16x32_f16 v[108:111], a[84:87], a[8:11], v[108:111]
	v_mfma_f32_16x16x32_f16 v[20:23], a[80:83], a[12:15], v[108:111]
	v_mfma_f32_16x16x32_f16 v[104:107], a[76:79], a[8:11], v[104:107]
	v_mfma_f32_16x16x32_f16 v[24:27], a[72:75], a[12:15], v[104:107]
	v_mfma_f32_16x16x32_f16 v[100:103], a[68:71], a[8:11], v[100:103]
	v_mfma_f32_16x16x32_f16 v[28:31], a[64:67], a[12:15], v[100:103]
	v_mfma_f32_16x16x32_f16 v[96:99], a[60:63], a[8:11], v[96:99]
	v_mfma_f32_16x16x32_f16 v[32:35], a[56:59], a[12:15], v[96:99]
	v_mfma_f32_16x16x32_f16 v[92:95], a[124:127], a[0:3], v[92:95]
	v_mfma_f32_16x16x32_f16 v[36:39], a[116:119], a[4:7], v[92:95]
	v_mfma_f32_16x16x32_f16 v[88:91], a[108:111], a[0:3], v[88:91]
	v_mfma_f32_16x16x32_f16 v[40:43], a[104:107], a[4:7], v[88:91]
	v_mfma_f32_16x16x32_f16 v[84:87], a[100:103], a[0:3], v[84:87]
	v_mfma_f32_16x16x32_f16 v[4:7], a[96:99], a[4:7], v[84:87]
	v_mfma_f32_16x16x32_f16 v[80:83], a[92:95], a[0:3], v[80:83]
	v_mfma_f32_16x16x32_f16 v[48:51], a[88:91], a[4:7], v[80:83]
	v_mfma_f32_16x16x32_f16 v[76:79], a[84:87], a[0:3], v[76:79]
	v_mfma_f32_16x16x32_f16 v[52:55], a[80:83], a[4:7], v[76:79]
	v_mfma_f32_16x16x32_f16 v[72:75], a[76:79], a[0:3], v[72:75]
	v_mfma_f32_16x16x32_f16 v[56:59], a[72:75], a[4:7], v[72:75]
	v_mfma_f32_16x16x32_f16 v[68:71], a[68:71], a[0:3], v[68:71]
	v_mfma_f32_16x16x32_f16 v[60:63], a[64:67], a[4:7], v[68:71]
	v_mfma_f32_16x16x32_f16 v[64:67], a[60:63], a[0:3], v[64:67]
	v_mfma_f32_16x16x32_f16 v[64:67], a[56:59], a[4:7], v[64:67]
	v_accvgpr_read_b32 v0, a184
	v_accvgpr_read_b32 v1, a185
	v_accvgpr_read_b32 v2, a186
	v_accvgpr_read_b32 v3, a187
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[68:71], a[124:127], a[120:123], v[0:3]
	v_mfma_f32_16x16x32_f16 v[68:71], a[116:119], a[112:115], v[68:71]
	s_nop 1
	v_accvgpr_read_b32 v0, a180
	v_accvgpr_read_b32 v1, a181
	v_accvgpr_read_b32 v2, a182
	v_accvgpr_read_b32 v3, a183
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[72:75], a[108:111], a[120:123], v[0:3]
	v_mfma_f32_16x16x32_f16 v[72:75], a[104:107], a[112:115], v[72:75]
	s_nop 1
	v_accvgpr_read_b32 v0, a176
	v_accvgpr_read_b32 v1, a177
	v_accvgpr_read_b32 v2, a178
	v_accvgpr_read_b32 v3, a179
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[76:79], a[100:103], a[120:123], v[0:3]
	v_mfma_f32_16x16x32_f16 v[76:79], a[96:99], a[112:115], v[76:79]
	s_nop 1
	v_accvgpr_read_b32 v0, a172
	v_accvgpr_read_b32 v1, a173
	v_accvgpr_read_b32 v2, a174
	v_accvgpr_read_b32 v3, a175
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[80:83], a[92:95], a[120:123], v[0:3]
	v_mfma_f32_16x16x32_f16 v[80:83], a[88:91], a[112:115], v[80:83]
	s_nop 1
	v_accvgpr_read_b32 v0, a168
	v_accvgpr_read_b32 v1, a169
	v_accvgpr_read_b32 v2, a170
	v_accvgpr_read_b32 v3, a171
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[84:87], a[84:87], a[120:123], v[0:3]
	v_mfma_f32_16x16x32_f16 v[84:87], a[80:83], a[112:115], v[84:87]
	s_nop 1
	v_accvgpr_read_b32 v0, a164
	v_accvgpr_read_b32 v1, a165
	v_accvgpr_read_b32 v2, a166
	v_accvgpr_read_b32 v3, a167
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[88:91], a[76:79], a[120:123], v[0:3]
	v_mfma_f32_16x16x32_f16 v[88:91], a[72:75], a[112:115], v[88:91]
	s_nop 1
	v_accvgpr_read_b32 v0, a160
	v_accvgpr_read_b32 v1, a161
	v_accvgpr_read_b32 v2, a162
	v_accvgpr_read_b32 v3, a163
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[92:95], a[68:71], a[120:123], v[0:3]
	v_mfma_f32_16x16x32_f16 v[92:95], a[64:67], a[112:115], v[92:95]
	s_nop 1
	v_accvgpr_read_b32 v0, a156
	v_accvgpr_read_b32 v1, a157
	v_accvgpr_read_b32 v2, a158
	v_accvgpr_read_b32 v3, a159
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[96:99], a[60:63], a[120:123], v[0:3]
	v_mfma_f32_16x16x32_f16 v[96:99], a[56:59], a[112:115], v[96:99]
	s_nop 1
	v_accvgpr_read_b32 v0, a152
	v_accvgpr_read_b32 v1, a153
	v_accvgpr_read_b32 v2, a154
	v_accvgpr_read_b32 v3, a155
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[100:103], a[124:127], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[100:103], a[116:119], a[48:51], v[100:103]
	s_nop 1
	v_accvgpr_read_b32 v0, a148
	v_accvgpr_read_b32 v1, a149
	v_accvgpr_read_b32 v2, a150
	v_accvgpr_read_b32 v3, a151
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[104:107], a[108:111], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[104:107], a[104:107], a[48:51], v[104:107]
	s_nop 1
	v_accvgpr_read_b32 v0, a144
	v_accvgpr_read_b32 v1, a145
	v_accvgpr_read_b32 v2, a146
	v_accvgpr_read_b32 v3, a147
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[108:111], a[100:103], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[108:111], a[96:99], a[48:51], v[108:111]
	s_nop 1
	v_accvgpr_read_b32 v0, a140
	v_accvgpr_read_b32 v1, a141
	v_accvgpr_read_b32 v2, a142
	v_accvgpr_read_b32 v3, a143
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[112:115], a[92:95], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[112:115], a[88:91], a[48:51], v[112:115]
	s_nop 1
	v_accvgpr_read_b32 v0, a136
	v_accvgpr_read_b32 v1, a137
	v_accvgpr_read_b32 v2, a138
	v_accvgpr_read_b32 v3, a139
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[116:119], a[84:87], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[116:119], a[80:83], a[48:51], v[116:119]
	s_nop 1
	v_accvgpr_read_b32 v0, a132
	v_accvgpr_read_b32 v1, a133
	v_accvgpr_read_b32 v2, a134
	v_accvgpr_read_b32 v3, a135
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[120:123], a[76:79], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[120:123], a[72:75], a[48:51], v[120:123]
	s_nop 1
	v_accvgpr_read_b32 v0, a128
	v_accvgpr_read_b32 v1, a129
	v_accvgpr_read_b32 v2, a130
	v_accvgpr_read_b32 v3, a131
	s_nop 1
	v_mfma_f32_16x16x32_f16 v[124:127], a[68:71], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[124:127], a[64:67], a[48:51], v[124:127]
	v_mfma_f32_16x16x32_f16 v[44:47], a[60:63], a[52:55], v[44:47]
	v_mfma_f32_16x16x32_f16 v[0:3], a[56:59], a[48:51], v[44:47]
	s_nop 6
	v_accvgpr_read_b32 v44, a188
	v_accvgpr_read_b32 v45, a189
	v_accvgpr_read_b32 v46, a190
	v_accvgpr_read_b32 v47, a191
	v_cvt_pk_f16_f32 v44, v44, v45
	v_cvt_pk_f16_f32 v45, v46, v47
	v_cvt_pk_f16_f32 v252, v248, v249
	v_cvt_pk_f16_f32 v253, v250, v251
	v_cvt_pk_f16_f32 v248, v244, v245
	v_cvt_pk_f16_f32 v249, v246, v247
	v_cvt_pk_f16_f32 v244, v240, v241
	v_cvt_pk_f16_f32 v245, v242, v243
	v_cvt_pk_f16_f32 v240, v236, v237
	v_cvt_pk_f16_f32 v241, v238, v239
	v_cvt_pk_f16_f32 v236, v232, v233
	v_cvt_pk_f16_f32 v237, v234, v235
	v_cvt_pk_f16_f32 v232, v228, v229
	v_cvt_pk_f16_f32 v233, v230, v231
	v_cvt_pk_f16_f32 v228, v224, v225
	v_cvt_pk_f16_f32 v229, v226, v227
	v_cvt_pk_f16_f32 v224, v220, v221
	v_cvt_pk_f16_f32 v225, v222, v223
	v_cvt_pk_f16_f32 v220, v216, v217
	v_cvt_pk_f16_f32 v221, v218, v219
	v_cvt_pk_f16_f32 v216, v212, v213
	v_cvt_pk_f16_f32 v217, v214, v215
	v_cvt_pk_f16_f32 v212, v208, v209
	v_cvt_pk_f16_f32 v213, v210, v211
	v_cvt_pk_f16_f32 v208, v204, v205
	v_cvt_pk_f16_f32 v209, v206, v207
	v_cvt_pk_f16_f32 v204, v200, v201
	v_cvt_pk_f16_f32 v205, v202, v203
	v_cvt_pk_f16_f32 v200, v196, v197
	v_cvt_pk_f16_f32 v201, v198, v199
	v_cvt_pk_f16_f32 v196, v192, v193
	v_cvt_pk_f16_f32 v197, v194, v195
	v_cvt_pk_f16_f32 v192, v188, v189
	v_cvt_pk_f16_f32 v193, v190, v191
	v_cvt_pk_f16_f32 v188, v184, v185
	v_cvt_pk_f16_f32 v189, v186, v187
	v_cvt_pk_f16_f32 v184, v180, v181
	v_cvt_pk_f16_f32 v185, v182, v183
	v_cvt_pk_f16_f32 v180, v176, v177
	v_cvt_pk_f16_f32 v181, v178, v179
	v_cvt_pk_f16_f32 v176, v172, v173
	v_cvt_pk_f16_f32 v177, v174, v175
	v_cvt_pk_f16_f32 v172, v168, v169
	v_cvt_pk_f16_f32 v173, v170, v171
	v_cvt_pk_f16_f32 v168, v164, v165
	v_cvt_pk_f16_f32 v169, v166, v167
	v_cvt_pk_f16_f32 v164, v160, v161
	v_cvt_pk_f16_f32 v165, v162, v163
	v_accvgpr_read_b32 v163, a27
	v_accvgpr_read_b32 v162, a26
	v_accvgpr_read_b32 v161, a25
	v_accvgpr_read_b32 v160, a24
	v_cvt_pk_f16_f32 v170, v160, v161
	v_cvt_pk_f16_f32 v171, v162, v163
	v_cvt_pk_f16_f32 v160, v156, v157
	v_cvt_pk_f16_f32 v161, v158, v159
	v_cvt_pk_f16_f32 v158, v152, v153
	v_cvt_pk_f16_f32 v159, v154, v155
	v_cvt_pk_f16_f32 v156, v148, v149
	v_cvt_pk_f16_f32 v157, v150, v151
	v_cvt_pk_f16_f32 v154, v144, v145
	v_cvt_pk_f16_f32 v155, v146, v147
	v_cvt_pk_f16_f32 v152, v140, v141
	v_cvt_pk_f16_f32 v153, v142, v143
	v_cvt_pk_f16_f32 v150, v136, v137
	v_cvt_pk_f16_f32 v151, v138, v139
	v_cvt_pk_f16_f32 v148, v132, v133
	v_cvt_pk_f16_f32 v149, v134, v135
	v_cvt_pk_f16_f32 v146, v128, v129
	v_cvt_pk_f16_f32 v147, v130, v131
	v_cvt_pk_f16_f32 v144, v8, v9
	v_cvt_pk_f16_f32 v145, v10, v11
	v_cvt_pk_f16_f32 v142, v12, v13
	v_cvt_pk_f16_f32 v143, v14, v15
	v_cvt_pk_f16_f32 v140, v16, v17
	v_cvt_pk_f16_f32 v141, v18, v19
	v_cvt_pk_f16_f32 v138, v20, v21
	v_cvt_pk_f16_f32 v139, v22, v23
	v_cvt_pk_f16_f32 v136, v24, v25
	v_cvt_pk_f16_f32 v137, v26, v27
	v_cvt_pk_f16_f32 v134, v28, v29
	v_cvt_pk_f16_f32 v135, v30, v31
	v_cvt_pk_f16_f32 v132, v32, v33
	v_cvt_pk_f16_f32 v133, v34, v35
	v_cvt_pk_f16_f32 v46, v36, v37
	v_cvt_pk_f16_f32 v47, v38, v39
	v_cvt_pk_f16_f32 v166, v40, v41
	v_cvt_pk_f16_f32 v167, v42, v43
	v_cvt_pk_f16_f32 v42, v4, v5
	v_cvt_pk_f16_f32 v43, v6, v7
	v_cvt_pk_f16_f32 v40, v48, v49
	v_cvt_pk_f16_f32 v41, v50, v51
	v_cvt_pk_f16_f32 v38, v52, v53
	v_cvt_pk_f16_f32 v39, v54, v55
	v_cvt_pk_f16_f32 v36, v56, v57
	v_cvt_pk_f16_f32 v37, v58, v59
	v_cvt_pk_f16_f32 v34, v60, v61
	v_cvt_pk_f16_f32 v35, v62, v63
	v_cvt_pk_f16_f32 v32, v64, v65
	v_cvt_pk_f16_f32 v33, v66, v67
	v_cvt_pk_f16_f32 v30, v68, v69
	v_cvt_pk_f16_f32 v31, v70, v71
	v_cvt_pk_f16_f32 v28, v72, v73
	v_cvt_pk_f16_f32 v29, v74, v75
	v_cvt_pk_f16_f32 v26, v76, v77
	v_cvt_pk_f16_f32 v27, v78, v79
	v_cvt_pk_f16_f32 v24, v80, v81
	v_cvt_pk_f16_f32 v25, v82, v83
	v_cvt_pk_f16_f32 v22, v84, v85
	v_cvt_pk_f16_f32 v23, v86, v87
	v_cvt_pk_f16_f32 v20, v88, v89
	v_cvt_pk_f16_f32 v21, v90, v91
	v_cvt_pk_f16_f32 v18, v92, v93
	v_cvt_pk_f16_f32 v19, v94, v95
	v_cvt_pk_f16_f32 v16, v96, v97
	v_cvt_pk_f16_f32 v17, v98, v99
	v_cvt_pk_f16_f32 v14, v100, v101
	v_cvt_pk_f16_f32 v15, v102, v103
	v_cvt_pk_f16_f32 v12, v104, v105
	v_cvt_pk_f16_f32 v13, v106, v107
	v_cvt_pk_f16_f32 v10, v108, v109
	v_cvt_pk_f16_f32 v11, v110, v111
	v_cvt_pk_f16_f32 v8, v112, v113
	v_cvt_pk_f16_f32 v9, v114, v115
	v_cvt_pk_f16_f32 v6, v116, v117
	v_cvt_pk_f16_f32 v7, v118, v119
	v_cvt_pk_f16_f32 v4, v120, v121
	v_cvt_pk_f16_f32 v5, v122, v123
	v_cvt_pk_f16_f32 v48, v124, v125
	v_cvt_pk_f16_f32 v49, v126, v127
	v_cvt_pk_f16_f32 v0, v0, v1
	v_cvt_pk_f16_f32 v1, v2, v3
	s_lshl_b32 s3, s3, 3
	v_accvgpr_read_b32 v2, a209
	v_and_or_b32 v2, s3, 16, v2
	v_or_b32_e32 v3, 32, v2
	v_or_b32_e32 v50, 64, v2
	v_or_b32_e32 v51, 0x60, v2
	v_or_b32_e32 v52, 0x80, v2
	v_or_b32_e32 v53, 0xa0, v2
	v_or_b32_e32 v54, 0xc0, v2
	v_or_b32_e32 v55, 0xe0, v2
	v_lshrrev_b32_e32 v56, 2, v254
	v_and_b32_e32 v56, 28, v56
	v_or_b32_e32 v57, 32, v56
	v_or_b32_e32 v58, 64, v56
	v_or_b32_e32 v59, 0x60, v56
	v_or_b32_e32 v131, 0x80, v56
	v_or_b32_e32 v130, 0xa0, v56
	v_or_b32_e32 v129, 0xc0, v56
	v_or_b32_e32 v128, 0xe0, v56
	s_mul_i32 s4, s0, s13
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_u32 s0, s6, s4
	s_addc_u32 s4, s7, s5
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s0, s0, s2
	s_addc_u32 s33, s4, s3
	v_mul_lo_u32 v60, v2, s13
	s_lshl_b32 s2, s13, 5
	v_add_u32_e32 v61, s2, v60
	v_add_u32_e32 v62, s2, v61
	v_add_u32_e32 v63, s2, v62
	v_add_u32_e32 v64, s2, v63
	v_add_u32_e32 v65, s2, v64
	v_add_u32_e32 v66, s2, v65
	v_add_u32_e32 v67, s2, v66
	v_cmp_gt_i32_e32 vcc, s8, v2
	v_cmp_gt_i32_e64 s[12:13], s8, v3
	v_cmp_gt_i32_e64 s[14:15], s8, v50
	v_cmp_gt_i32_e64 s[16:17], s8, v51
	v_cmp_gt_i32_e64 s[18:19], s8, v52
	v_cmp_gt_i32_e64 s[20:21], s8, v53
	v_cmp_gt_i32_e64 s[24:25], s8, v54
	v_cmp_gt_i32_e64 s[26:27], s8, v55
	v_cmp_gt_i32_e64 s[2:3], s1, v56
	v_cmp_gt_i32_e64 s[8:9], s1, v57
	v_cmp_gt_i32_e64 s[4:5], s1, v58
	v_cmp_gt_i32_e64 s[6:7], s1, v59
	v_cmp_gt_i32_e64 s[40:41], s1, v131
	v_cmp_gt_i32_e64 s[10:11], s1, v130
	v_cmp_gt_i32_e64 s[44:45], s1, v129
	v_cmp_gt_i32_e64 s[22:23], s1, v128
	s_and_b64 s[28:29], vcc, s[2:3]
	s_and_b64 s[30:31], vcc, s[8:9]
	s_and_b64 s[34:35], vcc, s[4:5]
	s_and_b64 s[36:37], vcc, s[6:7]
	s_and_b64 s[38:39], vcc, s[40:41]
	s_and_b64 s[42:43], vcc, s[10:11]
	s_and_b64 s[46:47], vcc, s[44:45]
	s_and_b64 vcc, vcc, s[22:23]
	s_and_b64 s[48:49], s[12:13], s[2:3]
	s_and_b64 s[50:51], s[12:13], s[8:9]
	s_and_b64 s[52:53], s[12:13], s[4:5]
	s_and_b64 s[54:55], s[12:13], s[6:7]
	s_and_b64 s[56:57], s[12:13], s[40:41]
	s_and_b64 s[58:59], s[12:13], s[10:11]
	s_and_b64 s[98:99], s[12:13], s[44:45]
	s_and_b64 s[96:97], s[12:13], s[22:23]
	s_and_b64 s[94:95], s[14:15], s[2:3]
	s_and_b64 s[92:93], s[14:15], s[8:9]
	s_and_b64 s[90:91], s[14:15], s[4:5]
	s_and_b64 s[88:89], s[14:15], s[6:7]
	s_and_b64 s[86:87], s[14:15], s[40:41]
	s_and_b64 s[84:85], s[14:15], s[10:11]
	s_and_b64 s[82:83], s[14:15], s[44:45]
	s_and_b64 s[80:81], s[14:15], s[22:23]
	s_and_b64 s[78:79], s[16:17], s[2:3]
	s_and_b64 s[76:77], s[16:17], s[8:9]
	s_and_b64 s[74:75], s[16:17], s[4:5]
	s_and_b64 s[72:73], s[16:17], s[6:7]
	s_and_b64 s[70:71], s[16:17], s[40:41]
	s_and_b64 s[68:69], s[16:17], s[10:11]
	s_and_b64 s[66:67], s[16:17], s[44:45]
	s_and_b64 s[64:65], s[16:17], s[22:23]
	s_and_b64 s[62:63], s[18:19], s[2:3]
	s_and_b64 s[60:61], s[18:19], s[8:9]
	s_and_b64 s[12:13], s[18:19], s[4:5]
	s_and_b64 s[14:15], s[18:19], s[6:7]
	s_and_b64 s[16:17], s[18:19], s[40:41]
                                        ; implicit-def: $vgpr255 : SGPR spill to VGPR lane
	v_writelane_b32 v255, s16, 0
	s_nop 1
	v_writelane_b32 v255, s17, 1
	s_and_b64 s[16:17], s[18:19], s[10:11]
	v_writelane_b32 v255, s16, 2
	s_nop 1
	v_writelane_b32 v255, s17, 3
	s_and_b64 s[16:17], s[18:19], s[44:45]
	v_writelane_b32 v255, s16, 4
	s_nop 1
	v_writelane_b32 v255, s17, 5
	s_and_b64 s[18:19], s[18:19], s[22:23]
	v_writelane_b32 v255, s18, 6
	s_nop 1
	v_writelane_b32 v255, s19, 7
	s_and_b64 s[18:19], s[20:21], s[2:3]
	v_writelane_b32 v255, s18, 8
	s_nop 1
	v_writelane_b32 v255, s19, 9
	s_and_b64 s[18:19], s[20:21], s[8:9]
	v_writelane_b32 v255, s18, 10
	s_nop 1
	v_writelane_b32 v255, s19, 11
	s_and_b64 s[18:19], s[20:21], s[4:5]
	v_writelane_b32 v255, s18, 12
	s_nop 1
	v_writelane_b32 v255, s19, 13
	s_and_b64 s[18:19], s[20:21], s[6:7]
	v_writelane_b32 v255, s18, 14
	s_nop 1
	v_writelane_b32 v255, s19, 15
	s_and_b64 s[18:19], s[20:21], s[40:41]
	v_writelane_b32 v255, s18, 16
	s_nop 1
	v_writelane_b32 v255, s19, 17
	s_and_b64 s[18:19], s[20:21], s[10:11]
	v_writelane_b32 v255, s18, 18
	s_nop 1
	v_writelane_b32 v255, s19, 19
	s_and_b64 s[18:19], s[20:21], s[44:45]
	v_writelane_b32 v255, s18, 20
	s_nop 1
	v_writelane_b32 v255, s19, 21
	s_and_b64 s[20:21], s[20:21], s[22:23]
	v_writelane_b32 v255, s20, 22
	s_nop 1
	v_writelane_b32 v255, s21, 23
	s_and_b64 s[20:21], s[24:25], s[2:3]
	v_writelane_b32 v255, s20, 24
	s_nop 1
	v_writelane_b32 v255, s21, 25
	s_and_b64 s[20:21], s[24:25], s[8:9]
	v_writelane_b32 v255, s20, 26
	s_nop 1
	v_writelane_b32 v255, s21, 27
	s_and_b64 s[20:21], s[24:25], s[4:5]
	v_writelane_b32 v255, s20, 28
	s_nop 1
	v_writelane_b32 v255, s21, 29
	s_and_b64 s[20:21], s[24:25], s[6:7]
	v_writelane_b32 v255, s20, 30
	s_nop 1
	v_writelane_b32 v255, s21, 31
	s_and_b64 s[20:21], s[24:25], s[40:41]
	v_writelane_b32 v255, s20, 32
	s_nop 1
	v_writelane_b32 v255, s21, 33
	s_and_b64 s[20:21], s[24:25], s[10:11]
	v_writelane_b32 v255, s20, 34
	s_nop 1
	v_writelane_b32 v255, s21, 35
	s_and_b64 s[20:21], s[24:25], s[44:45]
	v_writelane_b32 v255, s20, 36
	s_nop 1
	v_writelane_b32 v255, s21, 37
	s_and_b64 s[24:25], s[24:25], s[22:23]
	v_writelane_b32 v255, s24, 38
	s_nop 1
	v_writelane_b32 v255, s25, 39
	s_and_b64 s[2:3], s[26:27], s[2:3]
	v_writelane_b32 v255, s2, 40
	s_nop 1
	v_writelane_b32 v255, s3, 41
	s_and_b64 s[2:3], s[26:27], s[8:9]
	v_writelane_b32 v255, s2, 42
	s_nop 1
	v_writelane_b32 v255, s3, 43
	s_and_b64 s[2:3], s[26:27], s[4:5]
	v_writelane_b32 v255, s2, 44
	s_nop 1
	v_writelane_b32 v255, s3, 45
	s_and_b64 s[2:3], s[26:27], s[6:7]
	v_writelane_b32 v255, s2, 46
	s_nop 1
	v_writelane_b32 v255, s3, 47
	s_and_b64 s[24:25], s[26:27], s[40:41]
	s_and_b64 s[8:9], s[26:27], s[10:11]
	s_and_b64 s[6:7], s[26:27], s[44:45]
	s_and_b64 s[4:5], s[26:27], s[22:23]
	s_and_b32 s1, s33, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_add_lshl_u32 v2, v56, v60, 1
	v_bfrev_b32_e32 v3, 1
	v_cndmask_b32_e64 v2, v3, v2, s[28:29]
	buffer_store_dwordx2 v[44:45], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v60, 1
	v_cndmask_b32_e64 v2, v3, v2, s[30:31]
	buffer_store_dwordx2 v[252:253], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v60, 1
	v_cndmask_b32_e64 v2, v3, v2, s[34:35]
	buffer_store_dwordx2 v[248:249], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v60, 1
	v_cndmask_b32_e64 v2, v3, v2, s[36:37]
	buffer_store_dwordx2 v[244:245], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v131, v60, 1
	v_cndmask_b32_e64 v2, v3, v2, s[38:39]
	buffer_store_dwordx2 v[240:241], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v60, 1
	v_cndmask_b32_e64 v2, v3, v2, s[42:43]
	buffer_store_dwordx2 v[236:237], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v60, 1
	v_cndmask_b32_e64 v2, v3, v2, s[46:47]
	buffer_store_dwordx2 v[232:233], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v60, 1
	v_cndmask_b32_e32 v2, v3, v2, vcc
	buffer_store_dwordx2 v[228:229], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v61, v56, 1
	v_cndmask_b32_e64 v2, v3, v2, s[48:49]
	buffer_store_dwordx2 v[224:225], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[50:51]
	buffer_store_dwordx2 v[220:221], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[52:53]
	buffer_store_dwordx2 v[216:217], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[54:55]
	buffer_store_dwordx2 v[212:213], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v131, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[56:57]
	buffer_store_dwordx2 v[208:209], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[58:59]
	buffer_store_dwordx2 v[204:205], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[98:99]
	buffer_store_dwordx2 v[200:201], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v61, 1
	v_cndmask_b32_e64 v2, v3, v2, s[96:97]
	buffer_store_dwordx2 v[196:197], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v62, v56, 1
	v_cndmask_b32_e64 v2, v3, v2, s[94:95]
	buffer_store_dwordx2 v[192:193], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[92:93]
	buffer_store_dwordx2 v[188:189], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[90:91]
	buffer_store_dwordx2 v[184:185], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[88:89]
	buffer_store_dwordx2 v[180:181], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v131, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[86:87]
	buffer_store_dwordx2 v[176:177], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[84:85]
	buffer_store_dwordx2 v[172:173], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[82:83]
	buffer_store_dwordx2 v[168:169], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v62, 1
	v_cndmask_b32_e64 v2, v3, v2, s[80:81]
	buffer_store_dwordx2 v[164:165], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v63, v56, 1
	v_cndmask_b32_e64 v2, v3, v2, s[78:79]
	buffer_store_dwordx2 v[170:171], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[76:77]
	buffer_store_dwordx2 v[160:161], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[74:75]
	buffer_store_dwordx2 v[158:159], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[72:73]
	buffer_store_dwordx2 v[156:157], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v131, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[70:71]
	buffer_store_dwordx2 v[154:155], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[68:69]
	buffer_store_dwordx2 v[152:153], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[66:67]
	buffer_store_dwordx2 v[150:151], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v63, 1
	v_cndmask_b32_e64 v2, v3, v2, s[64:65]
	buffer_store_dwordx2 v[148:149], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v64, v56, 1
	v_cndmask_b32_e64 v2, v3, v2, s[62:63]
	buffer_store_dwordx2 v[146:147], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v64, 1
	v_cndmask_b32_e64 v2, v3, v2, s[60:61]
	buffer_store_dwordx2 v[144:145], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v64, 1
	v_cndmask_b32_e64 v2, v3, v2, s[12:13]
	buffer_store_dwordx2 v[142:143], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v64, 1
	v_cndmask_b32_e64 v2, v3, v2, s[14:15]
	buffer_store_dwordx2 v[140:141], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v131, v64, 1
	v_readlane_b32 s10, v255, 0
	v_readlane_b32 s11, v255, 1
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[138:139], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v64, 1
	v_readlane_b32 s10, v255, 2
	v_readlane_b32 s11, v255, 3
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[136:137], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v64, 1
	v_readlane_b32 s10, v255, 4
	v_readlane_b32 s11, v255, 5
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[134:135], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v64, 1
	v_readlane_b32 s10, v255, 6
	v_readlane_b32 s11, v255, 7
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[132:133], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v65, v56, 1
	v_readlane_b32 s10, v255, 8
	v_readlane_b32 s11, v255, 9
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[46:47], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v65, 1
	v_readlane_b32 s10, v255, 10
	v_readlane_b32 s11, v255, 11
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[166:167], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v65, 1
	v_readlane_b32 s10, v255, 12
	v_readlane_b32 s11, v255, 13
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[42:43], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v65, 1
	v_readlane_b32 s10, v255, 14
	v_readlane_b32 s11, v255, 15
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[40:41], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v131, v65, 1
	v_readlane_b32 s10, v255, 16
	v_readlane_b32 s11, v255, 17
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[38:39], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v65, 1
	v_readlane_b32 s10, v255, 18
	v_readlane_b32 s11, v255, 19
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[36:37], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v65, 1
	v_readlane_b32 s10, v255, 20
	v_readlane_b32 s11, v255, 21
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[34:35], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v65, 1
	v_readlane_b32 s10, v255, 22
	v_readlane_b32 s11, v255, 23
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[32:33], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v66, v56, 1
	v_readlane_b32 s10, v255, 24
	v_readlane_b32 s11, v255, 25
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[30:31], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v66, 1
	v_readlane_b32 s10, v255, 26
	v_readlane_b32 s11, v255, 27
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[28:29], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v66, 1
	v_readlane_b32 s10, v255, 28
	v_readlane_b32 s11, v255, 29
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[26:27], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v66, 1
	v_readlane_b32 s10, v255, 30
	v_readlane_b32 s11, v255, 31
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[24:25], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v131, v66, 1
	v_readlane_b32 s10, v255, 32
	v_readlane_b32 s11, v255, 33
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[22:23], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v66, 1
	v_readlane_b32 s10, v255, 34
	v_readlane_b32 s11, v255, 35
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[20:21], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v66, 1
	v_readlane_b32 s10, v255, 36
	v_readlane_b32 s11, v255, 37
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[18:19], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v66, 1
	v_readlane_b32 s10, v255, 38
	v_readlane_b32 s11, v255, 39
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[16:17], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v67, v56, 1
	v_readlane_b32 s10, v255, 40
	v_readlane_b32 s11, v255, 41
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[14:15], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v57, v67, 1
	v_readlane_b32 s10, v255, 42
	v_readlane_b32 s11, v255, 43
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[12:13], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v58, v67, 1
	v_readlane_b32 s10, v255, 44
	v_readlane_b32 s11, v255, 45
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[10:11], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v59, v67, 1
	v_readlane_b32 s10, v255, 46
	v_readlane_b32 s11, v255, 47
	s_nop 1
	v_cndmask_b32_e64 v2, v3, v2, s[10:11]
	buffer_store_dwordx2 v[8:9], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v131, v67, 1
	v_cndmask_b32_e64 v2, v3, v2, s[24:25]
	buffer_store_dwordx2 v[6:7], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v130, v67, 1
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[4:5], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v129, v67, 1
	v_cndmask_b32_e64 v2, v3, v2, s[6:7]
	buffer_store_dwordx2 v[48:49], v2, s[0:3], 0 offen
	v_add_lshl_u32 v2, v128, v67, 1
	v_cndmask_b32_e64 v2, v3, v2, s[4:5]
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v5_local_prefetch
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
		.amdhsa_next_free_sgpr 100
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
	.size	v5_local_prefetch, .Lfunc_end0-v5_local_prefetch
	.cfi_endproc
                                        ; -- End function
	.set v5_local_prefetch.num_vgpr, 256
	.set v5_local_prefetch.num_agpr, 256
	.set v5_local_prefetch.numbered_sgpr, 100
	.set v5_local_prefetch.num_named_barrier, 0
	.set v5_local_prefetch.private_seg_size, 0
	.set v5_local_prefetch.uses_vcc, 1
	.set v5_local_prefetch.uses_flat_scratch, 0
	.set v5_local_prefetch.has_dyn_sized_stack, 0
	.set v5_local_prefetch.has_recursion, 0
	.set v5_local_prefetch.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16308
; TotalNumSgprs: 106
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 106
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x58 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x32 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	54                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	146                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v5_local_prefetch" ; string offset=24
.Linfo_string3:
	.asciz	"v5_local_prefetch"             ; string offset=89
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
    .name:           v5_local_prefetch
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .sgpr_spill_count: 48
    .symbol:         v5_local_prefetch.kd
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
