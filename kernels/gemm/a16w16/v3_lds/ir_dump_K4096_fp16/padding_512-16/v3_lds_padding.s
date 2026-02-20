	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v3_lds_padding                  ; -- Begin function v3_lds_padding
	.p2align	8
	.type	v3_lds_padding,@function
v3_lds_padding:                         ; @v3_lds_padding
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.6:
	.file	1 "/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v3_lds" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.7:
.LBB0_0:
	.file	2 "/root/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s9, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s1, s0, 8
	s_abs_i32 s14, s1
	v_cvt_f32_u32_e32 v1, s14
	s_sub_i32 s19, 0, s14
	s_abs_i32 s18, s16
	v_readfirstlane_b32 s17, v0
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s15, s16, s1
	s_bfe_u32 s0, s17, 0x20006
	s_ashr_i32 s15, s15, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s20, v1
	s_mul_i32 s19, s19, s20
	s_mul_hi_u32 s19, s20, s19
	s_add_i32 s20, s20, s19
	s_mul_hi_u32 s19, s18, s20
	s_mul_i32 s20, s19, s14
	s_sub_i32 s18, s18, s20
	s_add_i32 s20, s19, 1
	s_sub_i32 s21, s18, s14
	s_cmp_ge_u32 s18, s14
	s_cselect_b32 s19, s20, s19
	s_cselect_b32 s18, s21, s18
	s_add_i32 s20, s19, 1
	s_cmp_ge_u32 s18, s14
	s_cselect_b32 s14, s20, s19
	s_xor_b32 s14, s14, s15
	s_sub_i32 s14, s14, s15
	s_mul_i32 s1, s14, s1
	s_sub_i32 s1, s16, s1
	v_and_b32_e32 v1, 63, v0
	s_lshl_b32 s15, s14, 8
	s_lshl_b32 s14, s1, 8
	s_add_i32 s1, s10, 63
	s_cmp_lt_i32 s1, 64
	v_lshl_or_b32 v19, s0, 6, v1
	s_cbranch_scc1 .LBB0_4
; %bb.1:                                ; %.lr.ph
	v_lshlrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v1, 0x70, v1
	v_mov_b32_e32 v18, v0
	v_or_b32_e32 v17, s0, v1
	v_lshlrev_b32_e32 v3, 3, v18
	v_and_b32_e32 v3, 56, v3
	v_or_b32_e32 v14, 4, v17
	v_mul_lo_u32 v0, v17, s11
	v_or_b32_e32 v12, 8, v17
	v_mul_lo_u32 v15, v14, s12
	v_mul_lo_u32 v14, v14, s11
	v_add_lshl_u32 v0, v0, v3, 1
	v_or_b32_e32 v10, 12, v17
	v_mul_lo_u32 v13, v12, s12
	v_mul_lo_u32 v12, v12, s11
	v_accvgpr_write_b32 a138, v0
	v_add_lshl_u32 v0, v14, v3, 1
	v_or_b32_e32 v8, 0x80, v17
	v_mul_lo_u32 v11, v10, s12
	v_mul_lo_u32 v10, v10, s11
	v_accvgpr_write_b32 a139, v0
	v_add_lshl_u32 v0, v12, v3, 1
	v_or_b32_e32 v6, 0x84, v17
	v_mul_lo_u32 v9, v8, s12
	v_mul_lo_u32 v8, v8, s11
	v_accvgpr_write_b32 a140, v0
	v_add_lshl_u32 v0, v10, v3, 1
	v_or_b32_e32 v4, 0x88, v17
	v_mul_lo_u32 v7, v6, s12
	v_mul_lo_u32 v6, v6, s11
	v_accvgpr_write_b32 a141, v0
	v_add_lshl_u32 v0, v8, v3, 1
	s_ashr_i32 s10, s1, 31
	v_or_b32_e32 v1, 0x8c, v17
	v_mul_lo_u32 v5, v4, s12
	v_mul_lo_u32 v4, v4, s11
	v_accvgpr_write_b32 a142, v0
	v_add_lshl_u32 v0, v6, v3, 1
	s_lshr_b32 s10, s10, 26
	v_mul_lo_u32 v2, v1, s12
	v_mul_lo_u32 v1, v1, s11
	s_mul_i32 s18, s14, s12
	v_accvgpr_write_b32 a143, v0
	v_add_lshl_u32 v0, v4, v3, 1
	s_add_i32 s1, s1, s10
	v_mul_lo_u32 v16, v17, s12
	s_ashr_i32 s19, s18, 31
	v_accvgpr_write_b32 a144, v0
	v_add_lshl_u32 v0, v1, v3, 1
	s_ashr_i32 s10, s1, 6
	s_lshl_b64 s[18:19], s[18:19], 1
	v_accvgpr_write_b32 a145, v0
	v_add_lshl_u32 v0, v16, v3, 1
	s_add_u32 s4, s4, s18
	s_mul_i32 s18, s15, s11
	v_accvgpr_write_b32 a146, v0
	v_add_lshl_u32 v0, v15, v3, 1
	s_addc_u32 s5, s5, s19
	s_ashr_i32 s19, s18, 31
	v_accvgpr_write_b32 a147, v0
	v_add_lshl_u32 v0, v13, v3, 1
	s_lshl_b64 s[18:19], s[18:19], 1
	v_accvgpr_write_b32 a148, v0
	v_add_lshl_u32 v0, v11, v3, 1
	s_add_u32 s11, s2, s18
	s_mulk_i32 s0, 0x420
	v_accvgpr_write_b32 a149, v0
	v_add_lshl_u32 v0, v9, v3, 1
	s_addc_u32 s12, s3, s19
	s_add_i32 s16, s0, 0
	v_accvgpr_write_b32 a150, v0
	v_add_lshl_u32 v0, v7, v3, 1
	s_lshl_b32 s0, s17, 1
	v_accvgpr_write_b32 a151, v0
	v_lshlrev_b32_e32 v0, 10, v18
	s_and_b32 s0, s0, 0x80
	v_mov_b32_e32 v250, 0
	v_add_lshl_u32 v254, v2, v3, 1
	v_and_b32_e32 v0, 0x3c00, v0
	v_and_b32_e32 v2, 0xb0, v19
	v_and_b32_e32 v4, 48, v18
	s_add_i32 s0, s0, 0
	v_mov_b32_e32 v9, v250
	v_add_lshl_u32 v1, v5, v3, 1
	v_add3_u32 v2, 0, v0, v2
	v_add3_u32 v4, s0, v4, v0
	v_lshrrev_b32_e32 v3, 5, v0
	v_mov_b32_e32 v6, v250
	v_mov_b32_e32 v7, v250
	v_mov_b32_e32 v8, v250
	v_accvgpr_write_b32 a137, v9
	s_add_i32 s18, s16, 0x1080
	s_add_i32 s19, s16, 0x2100
	s_add_i32 s20, s16, 0x3180
	s_add_i32 s21, s16, 0x4200
	v_accvgpr_write_b32 a133, v9
	s_add_i32 s22, s16, 0x5280
	s_add_i32 s23, s16, 0x6300
	s_add_i32 s24, s16, 0x7380
	s_add_i32 s25, s16, 0x83e0
	s_add_i32 s26, s16, 0x9460
	s_add_i32 s27, s16, 0xa4e0
	s_add_i32 s28, s16, 0xb560
	s_add_i32 s29, s16, 0xc5e0
	s_add_i32 s30, s16, 0xd660
	s_add_i32 s31, s16, 0xe6e0
	s_add_i32 s33, s16, 0xf760
	v_accvgpr_write_b32 a129, v19
	v_accvgpr_write_b32 a128, v18
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_add_u32_e32 v0, v2, v3
	v_add_u32_e32 v255, v4, v3
	v_mov_b32_e32 v251, v250
	v_mov_b32_e32 v252, v250
	v_mov_b32_e32 v253, v250
	v_mov_b32_e32 v2, v250
	v_mov_b32_e32 v3, v250
	v_mov_b32_e32 v4, v250
	v_mov_b32_e32 v5, v250
	v_mov_b32_e32 v206, v250
	v_mov_b32_e32 v207, v250
	v_mov_b32_e32 v208, v250
	v_mov_b32_e32 v209, v250
	v_mov_b32_e32 v242, v250
	v_mov_b32_e32 v243, v250
	v_mov_b32_e32 v244, v250
	v_mov_b32_e32 v245, v250
	v_mov_b32_e32 v238, v250
	v_mov_b32_e32 v239, v250
	v_mov_b32_e32 v240, v250
	v_mov_b32_e32 v241, v250
	v_mov_b32_e32 v234, v250
	v_mov_b32_e32 v235, v250
	v_mov_b32_e32 v236, v250
	v_mov_b32_e32 v237, v250
	v_mov_b32_e32 v230, v250
	v_mov_b32_e32 v231, v250
	v_mov_b32_e32 v232, v250
	v_mov_b32_e32 v233, v250
	v_mov_b32_e32 v226, v250
	v_mov_b32_e32 v227, v250
	v_mov_b32_e32 v228, v250
	v_mov_b32_e32 v229, v250
	v_mov_b32_e32 v222, v250
	v_mov_b32_e32 v223, v250
	v_mov_b32_e32 v224, v250
	v_mov_b32_e32 v225, v250
	v_mov_b32_e32 v218, v250
	v_mov_b32_e32 v219, v250
	v_mov_b32_e32 v220, v250
	v_mov_b32_e32 v221, v250
	v_mov_b32_e32 v214, v250
	v_mov_b32_e32 v215, v250
	v_mov_b32_e32 v216, v250
	v_mov_b32_e32 v217, v250
	v_accvgpr_write_b32 a132, v8
	v_accvgpr_write_b32 a131, v7
	v_accvgpr_write_b32 a130, v6
	v_mov_b32_e32 v202, v250
	v_mov_b32_e32 v203, v250
	v_mov_b32_e32 v204, v250
	v_mov_b32_e32 v205, v250
	v_mov_b32_e32 v198, v250
	v_mov_b32_e32 v199, v250
	v_mov_b32_e32 v200, v250
	v_mov_b32_e32 v201, v250
	v_mov_b32_e32 v194, v250
	v_mov_b32_e32 v195, v250
	v_mov_b32_e32 v196, v250
	v_mov_b32_e32 v197, v250
	v_mov_b32_e32 v190, v250
	v_mov_b32_e32 v191, v250
	v_mov_b32_e32 v192, v250
	v_mov_b32_e32 v193, v250
	v_mov_b32_e32 v186, v250
	v_mov_b32_e32 v187, v250
	v_mov_b32_e32 v188, v250
	v_mov_b32_e32 v189, v250
	v_mov_b32_e32 v182, v250
	v_mov_b32_e32 v183, v250
	v_mov_b32_e32 v184, v250
	v_mov_b32_e32 v185, v250
	v_mov_b32_e32 v178, v250
	v_mov_b32_e32 v179, v250
	v_mov_b32_e32 v180, v250
	v_mov_b32_e32 v181, v250
	v_mov_b32_e32 v174, v250
	v_mov_b32_e32 v175, v250
	v_mov_b32_e32 v176, v250
	v_mov_b32_e32 v177, v250
	v_mov_b32_e32 v170, v250
	v_mov_b32_e32 v171, v250
	v_mov_b32_e32 v172, v250
	v_mov_b32_e32 v173, v250
	v_mov_b32_e32 v166, v250
	v_mov_b32_e32 v167, v250
	v_mov_b32_e32 v168, v250
	v_mov_b32_e32 v169, v250
	v_mov_b32_e32 v162, v250
	v_mov_b32_e32 v163, v250
	v_mov_b32_e32 v164, v250
	v_mov_b32_e32 v165, v250
	v_mov_b32_e32 v158, v250
	v_mov_b32_e32 v159, v250
	v_mov_b32_e32 v160, v250
	v_mov_b32_e32 v161, v250
	v_mov_b32_e32 v154, v250
	v_mov_b32_e32 v155, v250
	v_mov_b32_e32 v156, v250
	v_mov_b32_e32 v157, v250
	v_mov_b32_e32 v150, v250
	v_mov_b32_e32 v151, v250
	v_mov_b32_e32 v152, v250
	v_mov_b32_e32 v153, v250
	v_mov_b32_e32 v146, v250
	v_mov_b32_e32 v147, v250
	v_mov_b32_e32 v148, v250
	v_mov_b32_e32 v149, v250
	v_mov_b32_e32 v142, v250
	v_mov_b32_e32 v143, v250
	v_mov_b32_e32 v144, v250
	v_mov_b32_e32 v145, v250
	v_mov_b32_e32 v138, v250
	v_mov_b32_e32 v139, v250
	v_mov_b32_e32 v140, v250
	v_mov_b32_e32 v141, v250
	v_mov_b32_e32 v134, v250
	v_mov_b32_e32 v135, v250
	v_mov_b32_e32 v136, v250
	v_mov_b32_e32 v137, v250
	v_mov_b32_e32 v122, v250
	v_mov_b32_e32 v123, v250
	v_mov_b32_e32 v124, v250
	v_mov_b32_e32 v125, v250
	v_mov_b32_e32 v118, v250
	v_mov_b32_e32 v119, v250
	v_mov_b32_e32 v120, v250
	v_mov_b32_e32 v121, v250
	v_mov_b32_e32 v114, v250
	v_mov_b32_e32 v115, v250
	v_mov_b32_e32 v116, v250
	v_mov_b32_e32 v117, v250
	v_mov_b32_e32 v110, v250
	v_mov_b32_e32 v111, v250
	v_mov_b32_e32 v112, v250
	v_mov_b32_e32 v113, v250
	v_mov_b32_e32 v106, v250
	v_mov_b32_e32 v107, v250
	v_mov_b32_e32 v108, v250
	v_mov_b32_e32 v109, v250
	v_mov_b32_e32 v102, v250
	v_mov_b32_e32 v103, v250
	v_mov_b32_e32 v104, v250
	v_mov_b32_e32 v105, v250
	v_mov_b32_e32 v98, v250
	v_mov_b32_e32 v99, v250
	v_mov_b32_e32 v100, v250
	v_mov_b32_e32 v101, v250
	v_mov_b32_e32 v94, v250
	v_mov_b32_e32 v95, v250
	v_mov_b32_e32 v96, v250
	v_mov_b32_e32 v97, v250
	v_mov_b32_e32 v90, v250
	v_mov_b32_e32 v91, v250
	v_mov_b32_e32 v92, v250
	v_mov_b32_e32 v93, v250
	v_mov_b32_e32 v86, v250
	v_mov_b32_e32 v87, v250
	v_mov_b32_e32 v88, v250
	v_mov_b32_e32 v89, v250
	v_mov_b32_e32 v126, v250
	v_mov_b32_e32 v127, v250
	v_mov_b32_e32 v128, v250
	v_mov_b32_e32 v129, v250
	v_mov_b32_e32 v82, v250
	v_mov_b32_e32 v83, v250
	v_mov_b32_e32 v84, v250
	v_mov_b32_e32 v85, v250
	v_mov_b32_e32 v78, v250
	v_mov_b32_e32 v79, v250
	v_mov_b32_e32 v80, v250
	v_mov_b32_e32 v81, v250
	v_mov_b32_e32 v74, v250
	v_mov_b32_e32 v75, v250
	v_mov_b32_e32 v76, v250
	v_mov_b32_e32 v77, v250
	v_mov_b32_e32 v70, v250
	v_mov_b32_e32 v71, v250
	v_mov_b32_e32 v72, v250
	v_mov_b32_e32 v73, v250
	v_mov_b32_e32 v66, v250
	v_mov_b32_e32 v67, v250
	v_mov_b32_e32 v68, v250
	v_mov_b32_e32 v69, v250
	v_mov_b32_e32 v62, v250
	v_mov_b32_e32 v63, v250
	v_mov_b32_e32 v64, v250
	v_mov_b32_e32 v65, v250
	v_mov_b32_e32 v54, v250
	v_mov_b32_e32 v55, v250
	v_mov_b32_e32 v56, v250
	v_mov_b32_e32 v57, v250
	v_accvgpr_write_b32 a160, v250
	v_accvgpr_write_b32 a161, v250
	v_accvgpr_write_b32 a162, v250
	v_accvgpr_write_b32 a163, v250
	v_mov_b32_e32 v38, v250
	v_mov_b32_e32 v39, v250
	v_mov_b32_e32 v40, v250
	v_mov_b32_e32 v41, v250
	v_accvgpr_write_b32 a136, v8
	v_accvgpr_write_b32 a135, v7
	v_accvgpr_write_b32 a134, v6
	v_mov_b32_e32 v246, v250
	v_mov_b32_e32 v247, v250
	v_mov_b32_e32 v248, v250
	v_mov_b32_e32 v249, v250
	v_mov_b32_e32 v34, v250
	v_mov_b32_e32 v35, v250
	v_mov_b32_e32 v36, v250
	v_mov_b32_e32 v37, v250
	v_mov_b32_e32 v50, v250
	v_mov_b32_e32 v51, v250
	v_mov_b32_e32 v52, v250
	v_mov_b32_e32 v53, v250
	v_mov_b32_e32 v46, v250
	v_mov_b32_e32 v47, v250
	v_mov_b32_e32 v48, v250
	v_mov_b32_e32 v49, v250
	v_mov_b32_e32 v42, v250
	v_mov_b32_e32 v43, v250
	v_mov_b32_e32 v44, v250
	v_mov_b32_e32 v45, v250
	v_mov_b32_e32 v30, v250
	v_mov_b32_e32 v31, v250
	v_mov_b32_e32 v32, v250
	v_mov_b32_e32 v33, v250
	v_accvgpr_write_b32 a156, v250
	v_accvgpr_write_b32 a157, v250
	v_accvgpr_write_b32 a158, v250
	v_accvgpr_write_b32 a159, v250
	v_mov_b32_e32 v18, v250
	v_mov_b32_e32 v19, v250
	v_mov_b32_e32 v20, v250
	v_mov_b32_e32 v21, v250
	v_mov_b32_e32 v14, v250
	v_mov_b32_e32 v15, v250
	v_mov_b32_e32 v16, v250
	v_mov_b32_e32 v17, v250
	v_mov_b32_e32 v10, v250
	v_mov_b32_e32 v11, v250
	v_mov_b32_e32 v12, v250
	v_mov_b32_e32 v13, v250
	v_accvgpr_write_b32 a152, v250
	v_accvgpr_write_b32 a153, v250
	v_accvgpr_write_b32 a154, v250
	v_accvgpr_write_b32 a155, v250
	v_mov_b32_e32 v130, v250
	v_mov_b32_e32 v131, v250
	v_mov_b32_e32 v132, v250
	v_mov_b32_e32 v133, v250
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	s_mov_b32 m0, s16
	s_and_b32 s1, s12, 0xffff
	s_mov_b32 s0, s11
	v_accvgpr_read_b32 v26, a138
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	v_accvgpr_read_b32 v26, a139
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s19
	v_accvgpr_read_b32 v26, a140
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	v_accvgpr_read_b32 v26, a141
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s21
	v_accvgpr_read_b32 v26, a142
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	v_accvgpr_read_b32 v26, a143
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s23
	v_accvgpr_read_b32 v26, a144
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	v_accvgpr_read_b32 v26, a145
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s25
	s_and_b32 s1, s5, 0xffff
	s_mov_b32 s0, s4
	v_accvgpr_read_b32 v26, a146
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	v_accvgpr_read_b32 v26, a147
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s27
	v_accvgpr_read_b32 v26, a148
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	v_accvgpr_read_b32 v26, a149
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s29
	v_accvgpr_read_b32 v26, a150
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	v_accvgpr_read_b32 v26, a151
	buffer_load_dwordx4 v26, s[0:3], 0 offen lds
	s_mov_b32 m0, s31
	s_add_u32 s11, s11, 0x80
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_addc_u32 s12, s12, 0
	buffer_load_dwordx4 v254, s[0:3], 0 offen lds
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b128 a[8:11], v255 offset:33760
	ds_read_b128 a[12:15], v255 offset:34016
	ds_read_b128 a[16:19], v255 offset:34272
	ds_read_b128 a[20:23], v255 offset:34528
	ds_read_b128 a[24:27], v255 offset:50656
	ds_read_b128 a[28:31], v255 offset:50912
	ds_read_b128 a[32:35], v255 offset:51168
	ds_read_b128 a[36:39], v255 offset:51424
	ds_read_b128 a[0:3], v255 offset:51488
	ds_read_b128 a[40:43], v0
	ds_read_b128 a[44:47], v0 offset:256
	ds_read_b128 a[48:51], v0 offset:512
	ds_read_b128 a[52:55], v0 offset:768
	ds_read_b128 a[56:59], v0 offset:16896
	ds_read_b128 a[60:63], v0 offset:17152
	ds_read_b128 a[64:67], v0 offset:17408
	ds_read_b128 a[68:71], v0 offset:17664
	ds_read_b128 a[72:75], v255 offset:33824
	ds_read_b128 a[76:79], v255 offset:34080
	ds_read_b128 a[80:83], v255 offset:34336
	ds_read_b128 a[84:87], v255 offset:34592
	ds_read_b128 a[88:91], v255 offset:50720
	ds_read_b128 a[92:95], v255 offset:50976
	ds_read_b128 a[96:99], v255 offset:51232
	ds_read_b128 a[100:103], v0 offset:64
	ds_read_b128 a[104:107], v0 offset:320
	ds_read_b128 a[108:111], v0 offset:576
	ds_read_b128 a[112:115], v0 offset:832
	ds_read_b128 a[116:119], v0 offset:16960
	ds_read_b128 a[120:123], v0 offset:17216
	ds_read_b128 a[124:127], v0 offset:17472
	ds_read_b128 a[4:7], v0 offset:17728
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[18:21], a[16:19], a[68:71], v[18:21]
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	s_add_i32 s10, s10, -1
	v_mfma_f32_16x16x32_f16 v[14:17], a[20:23], a[68:71], v[14:17]
	s_cmp_lg_u32 s10, 0
	v_mfma_f32_16x16x32_f16 v[10:13], a[24:27], a[68:71], v[10:13]
	v_mfma_f32_16x16x32_f16 v[6:9], a[28:31], a[68:71], v[6:9]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[26:29], a[80:83], a[4:7], v[18:21]
	v_mfma_f32_16x16x32_f16 v[18:21], a[84:87], a[4:7], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[88:91], a[4:7], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[92:95], a[4:7], v[6:9]
	v_mfma_f32_16x16x32_f16 v[6:9], a[8:11], a[68:71], v[30:33]
	v_mfma_f32_16x16x32_f16 v[30:33], a[72:75], a[4:7], v[6:9]
	v_mfma_f32_16x16x32_f16 v[6:9], a[20:23], a[64:67], v[246:249]
	v_mfma_f32_16x16x32_f16 v[246:249], a[84:87], a[124:127], v[6:9]
	v_mfma_f32_16x16x32_f16 v[6:9], a[24:27], a[64:67], v[34:37]
	v_mfma_f32_16x16x32_f16 v[250:253], a[8:11], a[40:43], v[250:253]
	v_mfma_f32_16x16x32_f16 v[38:41], a[12:15], a[64:67], v[38:41]
	v_mfma_f32_16x16x32_f16 a[152:155], a[32:35], a[68:71], a[152:155]
	v_mfma_f32_16x16x32_f16 v[130:133], a[36:39], a[68:71], v[130:133]
	v_mfma_f32_16x16x32_f16 a[134:137], a[16:19], a[64:67], a[134:137]
	v_mfma_f32_16x16x32_f16 a[160:163], a[8:11], a[64:67], a[160:163]
	v_mfma_f32_16x16x32_f16 v[126:129], a[8:11], a[60:63], v[126:129]
	v_mfma_f32_16x16x32_f16 a[68:71], a[12:15], a[68:71], a[156:159]
	v_mfma_f32_16x16x32_f16 v[2:5], a[12:15], a[40:43], v[2:5]
	v_mfma_f32_16x16x32_f16 v[206:209], a[16:19], a[40:43], v[206:209]
	v_mfma_f32_16x16x32_f16 v[242:245], a[20:23], a[40:43], v[242:245]
	v_mfma_f32_16x16x32_f16 v[238:241], a[24:27], a[40:43], v[238:241]
	v_mfma_f32_16x16x32_f16 v[234:237], a[28:31], a[40:43], v[234:237]
	v_mfma_f32_16x16x32_f16 v[230:233], a[32:35], a[40:43], v[230:233]
	v_mfma_f32_16x16x32_f16 v[226:229], a[36:39], a[40:43], v[226:229]
	v_mfma_f32_16x16x32_f16 v[222:225], a[8:11], a[44:47], v[222:225]
	v_mfma_f32_16x16x32_f16 v[218:221], a[12:15], a[44:47], v[218:221]
	v_mfma_f32_16x16x32_f16 v[214:217], a[16:19], a[44:47], v[214:217]
	v_mfma_f32_16x16x32_f16 a[130:133], a[20:23], a[44:47], a[130:133]
	v_mfma_f32_16x16x32_f16 v[202:205], a[24:27], a[44:47], v[202:205]
	v_mfma_f32_16x16x32_f16 v[198:201], a[28:31], a[44:47], v[198:201]
	v_mfma_f32_16x16x32_f16 v[194:197], a[32:35], a[44:47], v[194:197]
	v_mfma_f32_16x16x32_f16 v[190:193], a[36:39], a[44:47], v[190:193]
	v_mfma_f32_16x16x32_f16 v[186:189], a[8:11], a[48:51], v[186:189]
	v_mfma_f32_16x16x32_f16 v[182:185], a[12:15], a[48:51], v[182:185]
	v_mfma_f32_16x16x32_f16 v[178:181], a[16:19], a[48:51], v[178:181]
	v_mfma_f32_16x16x32_f16 v[174:177], a[20:23], a[48:51], v[174:177]
	v_mfma_f32_16x16x32_f16 v[170:173], a[24:27], a[48:51], v[170:173]
	v_mfma_f32_16x16x32_f16 v[166:169], a[28:31], a[48:51], v[166:169]
	v_mfma_f32_16x16x32_f16 v[162:165], a[32:35], a[48:51], v[162:165]
	v_mfma_f32_16x16x32_f16 v[158:161], a[36:39], a[48:51], v[158:161]
	v_mfma_f32_16x16x32_f16 v[154:157], a[8:11], a[52:55], v[154:157]
	v_mfma_f32_16x16x32_f16 v[150:153], a[12:15], a[52:55], v[150:153]
	v_mfma_f32_16x16x32_f16 v[146:149], a[16:19], a[52:55], v[146:149]
	v_mfma_f32_16x16x32_f16 v[142:145], a[20:23], a[52:55], v[142:145]
	v_mfma_f32_16x16x32_f16 v[138:141], a[24:27], a[52:55], v[138:141]
	v_mfma_f32_16x16x32_f16 v[134:137], a[28:31], a[52:55], v[134:137]
	v_mfma_f32_16x16x32_f16 v[122:125], a[32:35], a[52:55], v[122:125]
	v_mfma_f32_16x16x32_f16 v[118:121], a[36:39], a[52:55], v[118:121]
	v_mfma_f32_16x16x32_f16 v[114:117], a[8:11], a[56:59], v[114:117]
	v_mfma_f32_16x16x32_f16 v[110:113], a[12:15], a[56:59], v[110:113]
	v_mfma_f32_16x16x32_f16 v[106:109], a[16:19], a[56:59], v[106:109]
	v_mfma_f32_16x16x32_f16 v[102:105], a[20:23], a[56:59], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[24:27], a[56:59], v[98:101]
	v_mfma_f32_16x16x32_f16 v[94:97], a[28:31], a[56:59], v[94:97]
	v_mfma_f32_16x16x32_f16 v[90:93], a[32:35], a[56:59], v[90:93]
	v_mfma_f32_16x16x32_f16 v[86:89], a[36:39], a[56:59], v[86:89]
	v_mfma_f32_16x16x32_f16 v[82:85], a[12:15], a[60:63], v[82:85]
	v_mfma_f32_16x16x32_f16 v[78:81], a[16:19], a[60:63], v[78:81]
	v_mfma_f32_16x16x32_f16 v[74:77], a[20:23], a[60:63], v[74:77]
	v_mfma_f32_16x16x32_f16 v[70:73], a[24:27], a[60:63], v[70:73]
	v_mfma_f32_16x16x32_f16 v[66:69], a[28:31], a[60:63], v[66:69]
	v_mfma_f32_16x16x32_f16 v[62:65], a[32:35], a[60:63], v[62:65]
	v_mfma_f32_16x16x32_f16 v[54:57], a[36:39], a[60:63], v[54:57]
	v_mfma_f32_16x16x32_f16 v[50:53], a[28:31], a[64:67], v[50:53]
	v_mfma_f32_16x16x32_f16 v[46:49], a[32:35], a[64:67], v[46:49]
	v_mfma_f32_16x16x32_f16 v[42:45], a[36:39], a[64:67], v[42:45]
	v_mfma_f32_16x16x32_f16 v[34:37], a[88:91], a[124:127], v[6:9]
	s_nop 2
	v_mov_b64_e32 v[6:7], v[10:11]
	v_mfma_f32_16x16x32_f16 v[250:253], a[72:75], a[100:103], v[250:253]
	v_mov_b64_e32 v[8:9], v[12:13]
	v_mov_b64_e32 v[10:11], v[14:15]
	v_mov_b64_e32 v[12:13], v[16:17]
	v_mfma_f32_16x16x32_f16 v[38:41], a[76:79], a[124:127], v[38:41]
	v_mov_b64_e32 v[14:15], v[18:19]
	v_mov_b64_e32 v[16:17], v[20:21]
	v_mov_b64_e32 v[18:19], v[26:27]
	v_mfma_f32_16x16x32_f16 a[152:155], a[96:99], a[4:7], a[152:155]
	v_mov_b64_e32 v[20:21], v[28:29]
	v_mfma_f32_16x16x32_f16 a[134:137], a[80:83], a[124:127], a[134:137]
	v_mfma_f32_16x16x32_f16 v[126:129], a[72:75], a[120:123], v[126:129]
	v_mfma_f32_16x16x32_f16 a[156:159], a[76:79], a[4:7], a[68:71]
	v_mfma_f32_16x16x32_f16 v[2:5], a[76:79], a[100:103], v[2:5]
	v_mfma_f32_16x16x32_f16 v[206:209], a[80:83], a[100:103], v[206:209]
	v_mfma_f32_16x16x32_f16 v[242:245], a[84:87], a[100:103], v[242:245]
	v_mfma_f32_16x16x32_f16 v[238:241], a[88:91], a[100:103], v[238:241]
	v_mfma_f32_16x16x32_f16 v[234:237], a[92:95], a[100:103], v[234:237]
	v_mfma_f32_16x16x32_f16 v[230:233], a[96:99], a[100:103], v[230:233]
	v_mfma_f32_16x16x32_f16 v[226:229], a[0:3], a[100:103], v[226:229]
	v_mfma_f32_16x16x32_f16 v[222:225], a[72:75], a[104:107], v[222:225]
	v_mfma_f32_16x16x32_f16 v[218:221], a[76:79], a[104:107], v[218:221]
	v_mfma_f32_16x16x32_f16 v[214:217], a[80:83], a[104:107], v[214:217]
	v_mfma_f32_16x16x32_f16 a[130:133], a[84:87], a[104:107], a[130:133]
	v_mfma_f32_16x16x32_f16 v[202:205], a[88:91], a[104:107], v[202:205]
	v_mfma_f32_16x16x32_f16 v[198:201], a[92:95], a[104:107], v[198:201]
	v_mfma_f32_16x16x32_f16 v[194:197], a[96:99], a[104:107], v[194:197]
	v_mfma_f32_16x16x32_f16 v[190:193], a[0:3], a[104:107], v[190:193]
	v_mfma_f32_16x16x32_f16 v[186:189], a[72:75], a[108:111], v[186:189]
	v_mfma_f32_16x16x32_f16 v[182:185], a[76:79], a[108:111], v[182:185]
	v_mfma_f32_16x16x32_f16 v[178:181], a[80:83], a[108:111], v[178:181]
	v_mfma_f32_16x16x32_f16 v[174:177], a[84:87], a[108:111], v[174:177]
	v_mfma_f32_16x16x32_f16 v[170:173], a[88:91], a[108:111], v[170:173]
	v_mfma_f32_16x16x32_f16 v[166:169], a[92:95], a[108:111], v[166:169]
	v_mfma_f32_16x16x32_f16 v[162:165], a[96:99], a[108:111], v[162:165]
	v_mfma_f32_16x16x32_f16 v[158:161], a[0:3], a[108:111], v[158:161]
	v_mfma_f32_16x16x32_f16 v[154:157], a[72:75], a[112:115], v[154:157]
	v_mfma_f32_16x16x32_f16 v[150:153], a[76:79], a[112:115], v[150:153]
	v_mfma_f32_16x16x32_f16 v[146:149], a[80:83], a[112:115], v[146:149]
	v_mfma_f32_16x16x32_f16 v[142:145], a[84:87], a[112:115], v[142:145]
	v_mfma_f32_16x16x32_f16 v[138:141], a[88:91], a[112:115], v[138:141]
	v_mfma_f32_16x16x32_f16 v[134:137], a[92:95], a[112:115], v[134:137]
	v_mfma_f32_16x16x32_f16 v[122:125], a[96:99], a[112:115], v[122:125]
	v_mfma_f32_16x16x32_f16 v[118:121], a[0:3], a[112:115], v[118:121]
	v_mfma_f32_16x16x32_f16 v[114:117], a[72:75], a[116:119], v[114:117]
	v_mfma_f32_16x16x32_f16 v[110:113], a[76:79], a[116:119], v[110:113]
	v_mfma_f32_16x16x32_f16 v[106:109], a[80:83], a[116:119], v[106:109]
	v_mfma_f32_16x16x32_f16 v[102:105], a[84:87], a[116:119], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[88:91], a[116:119], v[98:101]
	v_mfma_f32_16x16x32_f16 v[94:97], a[92:95], a[116:119], v[94:97]
	v_mfma_f32_16x16x32_f16 v[90:93], a[96:99], a[116:119], v[90:93]
	v_mfma_f32_16x16x32_f16 v[86:89], a[0:3], a[116:119], v[86:89]
	v_mfma_f32_16x16x32_f16 v[82:85], a[76:79], a[120:123], v[82:85]
	v_mfma_f32_16x16x32_f16 v[78:81], a[80:83], a[120:123], v[78:81]
	v_mfma_f32_16x16x32_f16 v[74:77], a[84:87], a[120:123], v[74:77]
	v_mfma_f32_16x16x32_f16 v[70:73], a[88:91], a[120:123], v[70:73]
	v_mfma_f32_16x16x32_f16 v[66:69], a[92:95], a[120:123], v[66:69]
	v_mfma_f32_16x16x32_f16 v[62:65], a[96:99], a[120:123], v[62:65]
	v_mfma_f32_16x16x32_f16 v[54:57], a[0:3], a[120:123], v[54:57]
	v_mfma_f32_16x16x32_f16 a[160:163], a[72:75], a[124:127], a[160:163]
	v_mfma_f32_16x16x32_f16 v[50:53], a[92:95], a[124:127], v[50:53]
	v_mfma_f32_16x16x32_f16 v[46:49], a[96:99], a[124:127], v[46:49]
	v_mfma_f32_16x16x32_f16 v[42:45], a[0:3], a[124:127], v[42:45]
	v_mfma_f32_16x16x32_f16 v[130:133], a[0:3], a[4:7], v[130:133]
	s_cbranch_scc1 .LBB0_2
; %bb.3:                                ; %Flow
	v_mov_b64_e32 v[22:23], v[30:31]
	v_mov_b64_e32 v[26:27], v[34:35]
	s_nop 0
	v_accvgpr_mov_b32 a8, a160
	v_accvgpr_mov_b32 a4, a156
	v_accvgpr_mov_b32 a0, a152
	v_mov_b64_e32 v[24:25], v[32:33]
	v_mov_b64_e32 v[28:29], v[36:37]
	v_mov_b64_e32 v[30:31], v[246:247]
	v_accvgpr_read_b32 v34, a134
	v_accvgpr_read_b32 v58, a130
	v_accvgpr_mov_b32 a9, a161
	v_accvgpr_mov_b32 a10, a162
	v_accvgpr_mov_b32 a11, a163
	v_accvgpr_mov_b32 a5, a157
	v_accvgpr_mov_b32 a6, a158
	v_accvgpr_mov_b32 a7, a159
	v_accvgpr_mov_b32 a1, a153
	v_accvgpr_mov_b32 a2, a154
	v_accvgpr_mov_b32 a3, a155
	v_accvgpr_read_b32 v0, a128
	v_mov_b64_e32 v[32:33], v[248:249]
	v_accvgpr_read_b32 v35, a135
	v_accvgpr_read_b32 v36, a136
	v_accvgpr_read_b32 v37, a137
	v_accvgpr_read_b32 v59, a131
	v_accvgpr_read_b32 v60, a132
	v_accvgpr_read_b32 v61, a133
	s_branch .LBB0_5
.LBB0_4:
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v5, v133
	v_mov_b32_e32 v2, v133
	v_mov_b32_e32 v4, v133
	v_mov_b32_e32 v3, v133
	v_accvgpr_write_b32 a0, v2
	v_accvgpr_write_b32 a7, v5
	v_accvgpr_write_b32 a11, v5
	v_accvgpr_write_b32 a129, v19
	v_mov_b32_e32 v132, v133
	v_mov_b32_e32 v131, v133
	v_mov_b32_e32 v130, v133
	v_accvgpr_write_b32 a1, v3
	v_accvgpr_write_b32 a2, v4
	v_accvgpr_write_b32 a3, v5
	v_mov_b32_e32 v9, v133
	v_mov_b32_e32 v8, v133
	v_mov_b32_e32 v7, v133
	v_mov_b32_e32 v6, v133
	v_mov_b32_e32 v13, v133
	v_mov_b32_e32 v12, v133
	v_mov_b32_e32 v11, v133
	v_mov_b32_e32 v10, v133
	v_mov_b32_e32 v17, v133
	v_mov_b32_e32 v16, v133
	v_mov_b32_e32 v15, v133
	v_mov_b32_e32 v14, v133
	v_mov_b32_e32 v21, v133
	v_mov_b32_e32 v20, v133
	v_mov_b32_e32 v19, v133
	v_mov_b32_e32 v18, v133
	v_accvgpr_write_b32 a6, v4
	v_accvgpr_write_b32 a5, v3
	v_accvgpr_write_b32 a4, v2
	v_mov_b32_e32 v25, v133
	v_mov_b32_e32 v24, v133
	v_mov_b32_e32 v23, v133
	v_mov_b32_e32 v22, v133
	v_mov_b32_e32 v45, v133
	v_mov_b32_e32 v44, v133
	v_mov_b32_e32 v43, v133
	v_mov_b32_e32 v42, v133
	v_mov_b32_e32 v49, v133
	v_mov_b32_e32 v48, v133
	v_mov_b32_e32 v47, v133
	v_mov_b32_e32 v46, v133
	v_mov_b32_e32 v53, v133
	v_mov_b32_e32 v52, v133
	v_mov_b32_e32 v51, v133
	v_mov_b32_e32 v50, v133
	v_mov_b32_e32 v29, v133
	v_mov_b32_e32 v28, v133
	v_mov_b32_e32 v27, v133
	v_mov_b32_e32 v26, v133
	v_mov_b32_e32 v33, v133
	v_mov_b32_e32 v32, v133
	v_mov_b32_e32 v31, v133
	v_mov_b32_e32 v30, v133
	v_mov_b32_e32 v37, v133
	v_mov_b32_e32 v36, v133
	v_mov_b32_e32 v35, v133
	v_mov_b32_e32 v34, v133
	v_mov_b32_e32 v41, v133
	v_mov_b32_e32 v40, v133
	v_mov_b32_e32 v39, v133
	v_mov_b32_e32 v38, v133
	v_accvgpr_write_b32 a10, v4
	v_accvgpr_write_b32 a9, v3
	v_accvgpr_write_b32 a8, v2
	v_mov_b32_e32 v57, v133
	v_mov_b32_e32 v56, v133
	v_mov_b32_e32 v55, v133
	v_mov_b32_e32 v54, v133
	v_mov_b32_e32 v65, v133
	v_mov_b32_e32 v64, v133
	v_mov_b32_e32 v63, v133
	v_mov_b32_e32 v62, v133
	v_mov_b32_e32 v69, v133
	v_mov_b32_e32 v68, v133
	v_mov_b32_e32 v67, v133
	v_mov_b32_e32 v66, v133
	v_mov_b32_e32 v73, v133
	v_mov_b32_e32 v72, v133
	v_mov_b32_e32 v71, v133
	v_mov_b32_e32 v70, v133
	v_mov_b32_e32 v77, v133
	v_mov_b32_e32 v76, v133
	v_mov_b32_e32 v75, v133
	v_mov_b32_e32 v74, v133
	v_mov_b32_e32 v81, v133
	v_mov_b32_e32 v80, v133
	v_mov_b32_e32 v79, v133
	v_mov_b32_e32 v78, v133
	v_mov_b32_e32 v85, v133
	v_mov_b32_e32 v84, v133
	v_mov_b32_e32 v83, v133
	v_mov_b32_e32 v82, v133
	v_mov_b32_e32 v129, v133
	v_mov_b32_e32 v128, v133
	v_mov_b32_e32 v127, v133
	v_mov_b32_e32 v126, v133
	v_mov_b32_e32 v89, v133
	v_mov_b32_e32 v88, v133
	v_mov_b32_e32 v87, v133
	v_mov_b32_e32 v86, v133
	v_mov_b32_e32 v93, v133
	v_mov_b32_e32 v92, v133
	v_mov_b32_e32 v91, v133
	v_mov_b32_e32 v90, v133
	v_mov_b32_e32 v97, v133
	v_mov_b32_e32 v96, v133
	v_mov_b32_e32 v95, v133
	v_mov_b32_e32 v94, v133
	v_mov_b32_e32 v101, v133
	v_mov_b32_e32 v100, v133
	v_mov_b32_e32 v99, v133
	v_mov_b32_e32 v98, v133
	v_mov_b32_e32 v105, v133
	v_mov_b32_e32 v104, v133
	v_mov_b32_e32 v103, v133
	v_mov_b32_e32 v102, v133
	v_mov_b32_e32 v109, v133
	v_mov_b32_e32 v108, v133
	v_mov_b32_e32 v107, v133
	v_mov_b32_e32 v106, v133
	v_mov_b32_e32 v113, v133
	v_mov_b32_e32 v112, v133
	v_mov_b32_e32 v111, v133
	v_mov_b32_e32 v110, v133
	v_mov_b32_e32 v117, v133
	v_mov_b32_e32 v116, v133
	v_mov_b32_e32 v115, v133
	v_mov_b32_e32 v114, v133
	v_mov_b32_e32 v121, v133
	v_mov_b32_e32 v120, v133
	v_mov_b32_e32 v119, v133
	v_mov_b32_e32 v118, v133
	v_mov_b32_e32 v125, v133
	v_mov_b32_e32 v124, v133
	v_mov_b32_e32 v123, v133
	v_mov_b32_e32 v122, v133
	v_mov_b32_e32 v137, v133
	v_mov_b32_e32 v136, v133
	v_mov_b32_e32 v135, v133
	v_mov_b32_e32 v134, v133
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v140, v133
	v_mov_b32_e32 v139, v133
	v_mov_b32_e32 v138, v133
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v144, v133
	v_mov_b32_e32 v143, v133
	v_mov_b32_e32 v142, v133
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v148, v133
	v_mov_b32_e32 v147, v133
	v_mov_b32_e32 v146, v133
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v152, v133
	v_mov_b32_e32 v151, v133
	v_mov_b32_e32 v150, v133
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v156, v133
	v_mov_b32_e32 v155, v133
	v_mov_b32_e32 v154, v133
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v160, v133
	v_mov_b32_e32 v159, v133
	v_mov_b32_e32 v158, v133
	v_mov_b32_e32 v165, v133
	v_mov_b32_e32 v164, v133
	v_mov_b32_e32 v163, v133
	v_mov_b32_e32 v162, v133
	v_mov_b32_e32 v169, v133
	v_mov_b32_e32 v168, v133
	v_mov_b32_e32 v167, v133
	v_mov_b32_e32 v166, v133
	v_mov_b32_e32 v173, v133
	v_mov_b32_e32 v172, v133
	v_mov_b32_e32 v171, v133
	v_mov_b32_e32 v170, v133
	v_mov_b32_e32 v177, v133
	v_mov_b32_e32 v176, v133
	v_mov_b32_e32 v175, v133
	v_mov_b32_e32 v174, v133
	v_mov_b32_e32 v181, v133
	v_mov_b32_e32 v180, v133
	v_mov_b32_e32 v179, v133
	v_mov_b32_e32 v178, v133
	v_mov_b32_e32 v185, v133
	v_mov_b32_e32 v184, v133
	v_mov_b32_e32 v183, v133
	v_mov_b32_e32 v182, v133
	v_mov_b32_e32 v189, v133
	v_mov_b32_e32 v188, v133
	v_mov_b32_e32 v187, v133
	v_mov_b32_e32 v186, v133
	v_mov_b32_e32 v193, v133
	v_mov_b32_e32 v192, v133
	v_mov_b32_e32 v191, v133
	v_mov_b32_e32 v190, v133
	v_mov_b32_e32 v197, v133
	v_mov_b32_e32 v196, v133
	v_mov_b32_e32 v195, v133
	v_mov_b32_e32 v194, v133
	v_mov_b32_e32 v201, v133
	v_mov_b32_e32 v200, v133
	v_mov_b32_e32 v199, v133
	v_mov_b32_e32 v198, v133
	v_mov_b32_e32 v205, v133
	v_mov_b32_e32 v204, v133
	v_mov_b32_e32 v203, v133
	v_mov_b32_e32 v202, v133
	v_mov_b32_e32 v61, v133
	v_mov_b32_e32 v60, v133
	v_mov_b32_e32 v59, v133
	v_mov_b32_e32 v58, v133
	v_mov_b32_e32 v217, v133
	v_mov_b32_e32 v216, v133
	v_mov_b32_e32 v215, v133
	v_mov_b32_e32 v214, v133
	v_mov_b32_e32 v221, v133
	v_mov_b32_e32 v220, v133
	v_mov_b32_e32 v219, v133
	v_mov_b32_e32 v218, v133
	v_mov_b32_e32 v225, v133
	v_mov_b32_e32 v224, v133
	v_mov_b32_e32 v223, v133
	v_mov_b32_e32 v222, v133
	v_mov_b32_e32 v229, v133
	v_mov_b32_e32 v228, v133
	v_mov_b32_e32 v227, v133
	v_mov_b32_e32 v226, v133
	v_mov_b32_e32 v233, v133
	v_mov_b32_e32 v232, v133
	v_mov_b32_e32 v231, v133
	v_mov_b32_e32 v230, v133
	v_mov_b32_e32 v237, v133
	v_mov_b32_e32 v236, v133
	v_mov_b32_e32 v235, v133
	v_mov_b32_e32 v234, v133
	v_mov_b32_e32 v241, v133
	v_mov_b32_e32 v240, v133
	v_mov_b32_e32 v239, v133
	v_mov_b32_e32 v238, v133
	v_mov_b32_e32 v245, v133
	v_mov_b32_e32 v244, v133
	v_mov_b32_e32 v243, v133
	v_mov_b32_e32 v242, v133
	v_mov_b32_e32 v209, v133
	v_mov_b32_e32 v208, v133
	v_mov_b32_e32 v207, v133
	v_mov_b32_e32 v206, v133
	v_mov_b32_e32 v253, v133
	v_mov_b32_e32 v252, v133
	v_mov_b32_e32 v251, v133
	v_mov_b32_e32 v250, v133
.LBB0_5:                                ; %Flow318
	s_lshr_b32 s0, s17, 6
	v_and_b32_e32 v0, 15, v0
	s_lshl_b32 s0, s0, 3
	v_and_or_b32 v0, s0, 16, v0
	s_mul_i32 s0, s15, s13
	s_ashr_i32 s1, s0, 31
	v_cvt_pk_f16_f32 v255, v252, v253
	v_cvt_pk_f16_f32 v252, v2, v3
	v_cvt_pk_f16_f32 v253, v4, v5
	v_accvgpr_read_b32 v2, a8
	s_lshl_b64 s[0:1], s[0:1], 1
	v_accvgpr_read_b32 v3, a9
	v_accvgpr_read_b32 v4, a10
	v_accvgpr_read_b32 v5, a11
	v_cvt_pk_f16_f32 v18, v18, v19
	v_cvt_pk_f16_f32 v19, v20, v21
	v_accvgpr_read_b32 v20, a129
	s_add_u32 s2, s6, s0
	v_cvt_pk_f16_f32 v246, v242, v243
	v_cvt_pk_f16_f32 v242, v238, v239
	v_cvt_pk_f16_f32 v238, v234, v235
	v_cvt_pk_f16_f32 v234, v230, v231
	v_cvt_pk_f16_f32 v230, v226, v227
	v_cvt_pk_f16_f32 v226, v222, v223
	v_cvt_pk_f16_f32 v222, v218, v219
	v_cvt_pk_f16_f32 v218, v214, v215
	v_cvt_pk_f16_f32 v214, v58, v59
	v_cvt_pk_f16_f32 v58, v54, v55
	v_cvt_pk_f16_f32 v59, v56, v57
	v_cvt_pk_f16_f32 v56, v2, v3
	v_cvt_pk_f16_f32 v57, v4, v5
	v_cvt_pk_f16_f32 v54, v38, v39
	v_cvt_pk_f16_f32 v55, v40, v41
	v_cvt_pk_f16_f32 v38, v34, v35
	v_cvt_pk_f16_f32 v35, v28, v29
	v_cvt_pk_f16_f32 v28, v42, v43
	v_accvgpr_read_b32 v2, a4
	v_accvgpr_read_b32 v43, a3
	v_lshrrev_b32_e32 v20, 2, v20
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s15, s14, 31
	v_accvgpr_read_b32 v4, a6
	v_accvgpr_read_b32 v5, a7
	v_accvgpr_read_b32 v42, a2
	v_accvgpr_read_b32 v41, a1
	v_accvgpr_read_b32 v40, a0
	v_and_b32_e32 v20, 28, v20
	s_lshl_b64 s[0:1], s[14:15], 1
	v_cvt_pk_f16_f32 v34, v26, v27
	v_cvt_pk_f16_f32 v29, v44, v45
	v_cvt_pk_f16_f32 v26, v22, v23
	v_cvt_pk_f16_f32 v27, v24, v25
	v_cvt_pk_f16_f32 v23, v4, v5
	v_cvt_pk_f16_f32 v14, v14, v15
	v_cvt_pk_f16_f32 v15, v16, v17
	v_cvt_pk_f16_f32 v10, v10, v11
	v_cvt_pk_f16_f32 v11, v12, v13
	v_cvt_pk_f16_f32 v6, v6, v7
	v_cvt_pk_f16_f32 v7, v8, v9
	v_cvt_pk_f16_f32 v4, v40, v41
	v_cvt_pk_f16_f32 v5, v42, v43
	v_or_b32_e32 v1, 32, v0
	v_or_b32_e32 v8, 64, v0
	v_or_b32_e32 v9, 0x60, v0
	v_or_b32_e32 v12, 0x80, v0
	v_or_b32_e32 v13, 0xa0, v0
	v_or_b32_e32 v16, 0xc0, v0
	v_or_b32_e32 v17, 0xe0, v0
	v_or_b32_e32 v21, 32, v20
	v_or_b32_e32 v24, 64, v20
	v_or_b32_e32 v25, 0x60, v20
	v_or_b32_e32 v40, 0x80, v20
	v_or_b32_e32 v41, 0xa0, v20
	v_or_b32_e32 v42, 0xc0, v20
	v_or_b32_e32 v43, 0xe0, v20
	s_add_u32 s36, s2, s0
	v_mul_lo_u32 v44, v0, s13
	v_cmp_gt_i32_e64 s[28:29], s8, v0
	v_cmp_gt_i32_e64 s[14:15], s9, v20
	v_cvt_pk_f16_f32 v39, v36, v37
	v_cvt_pk_f16_f32 v36, v30, v31
	v_cvt_pk_f16_f32 v37, v32, v33
	v_cvt_pk_f16_f32 v32, v50, v51
	v_cvt_pk_f16_f32 v30, v46, v47
	v_cvt_pk_f16_f32 v31, v48, v49
	s_addc_u32 s33, s3, s1
	v_mul_lo_u32 v45, v1, s13
	v_mul_lo_u32 v46, v8, s13
	v_mul_lo_u32 v47, v9, s13
	v_mul_lo_u32 v48, v12, s13
	v_mul_lo_u32 v49, v13, s13
	v_mul_lo_u32 v50, v16, s13
	v_mul_lo_u32 v51, v17, s13
	v_cmp_gt_i32_e64 s[26:27], s8, v1
	v_cmp_gt_i32_e64 s[24:25], s8, v8
	v_cmp_gt_i32_e64 s[22:23], s8, v9
	v_cmp_gt_i32_e64 s[20:21], s8, v12
	v_cmp_gt_i32_e64 s[18:19], s8, v13
	v_cmp_gt_i32_e64 s[16:17], s8, v16
	v_cmp_gt_i32_e32 vcc, s8, v17
	v_cmp_gt_i32_e64 s[12:13], s9, v21
	v_cmp_gt_i32_e64 s[10:11], s9, v24
	v_cmp_gt_i32_e64 s[30:31], s9, v25
	v_cmp_gt_i32_e64 s[6:7], s9, v40
	v_cmp_gt_i32_e64 s[4:5], s9, v41
	v_cmp_gt_i32_e64 s[2:3], s9, v42
	v_cmp_gt_i32_e64 s[0:1], s9, v43
	v_add_lshl_u32 v0, v20, v44, 1
	v_bfrev_b32_e32 v1, 1
	s_and_b64 s[8:9], s[28:29], s[14:15]
	v_cvt_pk_f16_f32 v254, v250, v251
	s_and_b32 s37, s33, 0xffff
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[254:255], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v44, 1
	s_and_b64 s[8:9], s[28:29], s[12:13]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[252:253], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v44, 1
	s_and_b64 s[8:9], s[28:29], s[10:11]
	v_cvt_pk_f16_f32 v250, v206, v207
	v_cvt_pk_f16_f32 v251, v208, v209
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[250:251], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v44, 1
	s_and_b64 s[8:9], s[28:29], s[30:31]
	v_cvt_pk_f16_f32 v247, v244, v245
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[246:247], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v40, v44, 1
	s_and_b64 s[8:9], s[28:29], s[6:7]
	v_cvt_pk_f16_f32 v243, v240, v241
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[242:243], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v41, v44, 1
	s_and_b64 s[8:9], s[28:29], s[4:5]
	v_cvt_pk_f16_f32 v239, v236, v237
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[238:239], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v42, v44, 1
	s_and_b64 s[8:9], s[28:29], s[2:3]
	v_cvt_pk_f16_f32 v235, v232, v233
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[234:235], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v43, v44, 1
	s_and_b64 s[8:9], s[28:29], s[0:1]
	v_cvt_pk_f16_f32 v231, v228, v229
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[230:231], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v45, v20, 1
	s_and_b64 s[8:9], s[26:27], s[14:15]
	v_cvt_pk_f16_f32 v227, v224, v225
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[226:227], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v45, 1
	s_and_b64 s[8:9], s[26:27], s[12:13]
	v_cvt_pk_f16_f32 v223, v220, v221
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[222:223], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v45, 1
	s_and_b64 s[8:9], s[26:27], s[10:11]
	v_cvt_pk_f16_f32 v219, v216, v217
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[218:219], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v45, 1
	s_and_b64 s[8:9], s[26:27], s[30:31]
	v_cvt_pk_f16_f32 v215, v60, v61
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[214:215], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v40, v45, 1
	s_and_b64 s[8:9], s[26:27], s[6:7]
	v_cvt_pk_f16_f32 v210, v202, v203
	v_cvt_pk_f16_f32 v211, v204, v205
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[210:211], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v41, v45, 1
	s_and_b64 s[8:9], s[26:27], s[4:5]
	v_cvt_pk_f16_f32 v206, v198, v199
	v_cvt_pk_f16_f32 v207, v200, v201
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[206:207], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v42, v45, 1
	s_and_b64 s[8:9], s[26:27], s[2:3]
	v_cvt_pk_f16_f32 v198, v194, v195
	v_cvt_pk_f16_f32 v199, v196, v197
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[198:199], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v43, v45, 1
	s_and_b64 s[8:9], s[26:27], s[0:1]
	v_cvt_pk_f16_f32 v194, v190, v191
	v_cvt_pk_f16_f32 v195, v192, v193
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[194:195], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v46, v20, 1
	s_and_b64 s[8:9], s[24:25], s[14:15]
	v_cvt_pk_f16_f32 v190, v186, v187
	v_cvt_pk_f16_f32 v191, v188, v189
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[190:191], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v46, 1
	s_and_b64 s[8:9], s[24:25], s[12:13]
	v_cvt_pk_f16_f32 v186, v182, v183
	v_cvt_pk_f16_f32 v187, v184, v185
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[186:187], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v46, 1
	s_and_b64 s[8:9], s[24:25], s[10:11]
	v_cvt_pk_f16_f32 v182, v178, v179
	v_cvt_pk_f16_f32 v183, v180, v181
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[182:183], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v46, 1
	s_and_b64 s[8:9], s[24:25], s[30:31]
	v_cvt_pk_f16_f32 v178, v174, v175
	v_cvt_pk_f16_f32 v179, v176, v177
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[178:179], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v40, v46, 1
	s_and_b64 s[8:9], s[24:25], s[6:7]
	v_cvt_pk_f16_f32 v174, v170, v171
	v_cvt_pk_f16_f32 v175, v172, v173
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[174:175], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v41, v46, 1
	s_and_b64 s[8:9], s[24:25], s[4:5]
	v_cvt_pk_f16_f32 v170, v166, v167
	v_cvt_pk_f16_f32 v171, v168, v169
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[170:171], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v42, v46, 1
	s_and_b64 s[8:9], s[24:25], s[2:3]
	v_cvt_pk_f16_f32 v166, v162, v163
	v_cvt_pk_f16_f32 v167, v164, v165
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[166:167], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v43, v46, 1
	s_and_b64 s[8:9], s[24:25], s[0:1]
	v_cvt_pk_f16_f32 v162, v158, v159
	v_cvt_pk_f16_f32 v163, v160, v161
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[162:163], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v47, v20, 1
	s_and_b64 s[8:9], s[22:23], s[14:15]
	v_cvt_pk_f16_f32 v158, v154, v155
	v_cvt_pk_f16_f32 v159, v156, v157
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[158:159], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v47, 1
	s_and_b64 s[8:9], s[22:23], s[12:13]
	v_cvt_pk_f16_f32 v154, v150, v151
	v_cvt_pk_f16_f32 v155, v152, v153
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[154:155], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v47, 1
	s_and_b64 s[8:9], s[22:23], s[10:11]
	v_cvt_pk_f16_f32 v150, v146, v147
	v_cvt_pk_f16_f32 v151, v148, v149
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[150:151], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v47, 1
	s_and_b64 s[8:9], s[22:23], s[30:31]
	v_cvt_pk_f16_f32 v146, v142, v143
	v_cvt_pk_f16_f32 v147, v144, v145
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[146:147], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v40, v47, 1
	s_and_b64 s[8:9], s[22:23], s[6:7]
	v_cvt_pk_f16_f32 v142, v138, v139
	v_cvt_pk_f16_f32 v143, v140, v141
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[142:143], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v41, v47, 1
	s_and_b64 s[8:9], s[22:23], s[4:5]
	v_cvt_pk_f16_f32 v138, v134, v135
	v_cvt_pk_f16_f32 v139, v136, v137
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[138:139], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v42, v47, 1
	s_and_b64 s[8:9], s[22:23], s[2:3]
	v_cvt_pk_f16_f32 v134, v122, v123
	v_cvt_pk_f16_f32 v135, v124, v125
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[134:135], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v43, v47, 1
	s_and_b64 s[8:9], s[22:23], s[0:1]
	v_cvt_pk_f16_f32 v122, v118, v119
	v_cvt_pk_f16_f32 v123, v120, v121
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[122:123], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v48, v20, 1
	s_and_b64 s[8:9], s[20:21], s[14:15]
	v_cvt_pk_f16_f32 v118, v114, v115
	v_cvt_pk_f16_f32 v119, v116, v117
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[118:119], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v48, 1
	s_and_b64 s[8:9], s[20:21], s[12:13]
	v_cvt_pk_f16_f32 v114, v110, v111
	v_cvt_pk_f16_f32 v115, v112, v113
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[114:115], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v48, 1
	s_and_b64 s[8:9], s[20:21], s[10:11]
	v_cvt_pk_f16_f32 v110, v106, v107
	v_cvt_pk_f16_f32 v111, v108, v109
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[110:111], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v48, 1
	s_and_b64 s[8:9], s[20:21], s[30:31]
	v_cvt_pk_f16_f32 v106, v102, v103
	v_cvt_pk_f16_f32 v107, v104, v105
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[106:107], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v40, v48, 1
	s_and_b64 s[8:9], s[20:21], s[6:7]
	v_cvt_pk_f16_f32 v102, v98, v99
	v_cvt_pk_f16_f32 v103, v100, v101
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[102:103], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v41, v48, 1
	s_and_b64 s[8:9], s[20:21], s[4:5]
	v_cvt_pk_f16_f32 v98, v94, v95
	v_cvt_pk_f16_f32 v99, v96, v97
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[98:99], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v42, v48, 1
	s_and_b64 s[8:9], s[20:21], s[2:3]
	v_cvt_pk_f16_f32 v94, v90, v91
	v_cvt_pk_f16_f32 v95, v92, v93
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[94:95], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v43, v48, 1
	s_and_b64 s[8:9], s[20:21], s[0:1]
	v_cvt_pk_f16_f32 v90, v86, v87
	v_cvt_pk_f16_f32 v91, v88, v89
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[90:91], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v49, v20, 1
	s_and_b64 s[8:9], s[18:19], s[14:15]
	v_cvt_pk_f16_f32 v86, v126, v127
	v_cvt_pk_f16_f32 v87, v128, v129
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[86:87], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v49, 1
	s_and_b64 s[8:9], s[18:19], s[12:13]
	v_cvt_pk_f16_f32 v82, v82, v83
	v_cvt_pk_f16_f32 v83, v84, v85
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[82:83], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v49, 1
	s_and_b64 s[8:9], s[18:19], s[10:11]
	v_cvt_pk_f16_f32 v78, v78, v79
	v_cvt_pk_f16_f32 v79, v80, v81
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[78:79], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v49, 1
	s_and_b64 s[8:9], s[18:19], s[30:31]
	v_cvt_pk_f16_f32 v74, v74, v75
	v_cvt_pk_f16_f32 v75, v76, v77
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[74:75], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v40, v49, 1
	s_and_b64 s[8:9], s[18:19], s[6:7]
	v_cvt_pk_f16_f32 v70, v70, v71
	v_cvt_pk_f16_f32 v71, v72, v73
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[70:71], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v41, v49, 1
	s_and_b64 s[8:9], s[18:19], s[4:5]
	v_cvt_pk_f16_f32 v66, v66, v67
	v_cvt_pk_f16_f32 v67, v68, v69
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[66:67], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v42, v49, 1
	s_and_b64 s[8:9], s[18:19], s[2:3]
	v_cvt_pk_f16_f32 v60, v62, v63
	v_cvt_pk_f16_f32 v61, v64, v65
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[60:61], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v43, v49, 1
	s_and_b64 s[8:9], s[18:19], s[0:1]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[58:59], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v50, v20, 1
	s_and_b64 s[8:9], s[16:17], s[14:15]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[56:57], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v50, 1
	s_and_b64 s[8:9], s[16:17], s[12:13]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[54:55], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v50, 1
	s_and_b64 s[8:9], s[16:17], s[10:11]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[38:39], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v50, 1
	s_and_b64 s[8:9], s[16:17], s[30:31]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[36:37], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v40, v50, 1
	s_and_b64 s[8:9], s[16:17], s[6:7]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[34:35], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v41, v50, 1
	s_and_b64 s[8:9], s[16:17], s[4:5]
	v_cvt_pk_f16_f32 v33, v52, v53
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[32:33], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v42, v50, 1
	s_and_b64 s[8:9], s[16:17], s[2:3]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[30:31], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v43, v50, 1
	s_and_b64 s[8:9], s[16:17], s[0:1]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[28:29], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v51, v20, 1
	s_and_b64 s[8:9], vcc, s[14:15]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	v_accvgpr_read_b32 v3, a5
	buffer_store_dwordx2 v[26:27], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v51, 1
	s_and_b64 s[8:9], vcc, s[12:13]
	v_cvt_pk_f16_f32 v22, v2, v3
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[22:23], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v51, 1
	s_and_b64 s[8:9], vcc, s[10:11]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[18:19], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v51, 1
	s_and_b64 s[8:9], vcc, s[30:31]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[14:15], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v40, v51, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_cndmask_b32_e64 v0, v1, v0, s[6:7]
	buffer_store_dwordx2 v[10:11], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v41, v51, 1
	s_and_b64 s[4:5], vcc, s[4:5]
	v_cndmask_b32_e64 v0, v1, v0, s[4:5]
	buffer_store_dwordx2 v[6:7], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v42, v51, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_cndmask_b32_e64 v0, v1, v0, s[2:3]
	buffer_store_dwordx2 v[4:5], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v43, v51, 1
	s_and_b64 vcc, vcc, s[0:1]
	v_cvt_pk_f16_f32 v2, v130, v131
	v_cvt_pk_f16_f32 v3, v132, v133
	v_cndmask_b32_e32 v0, v1, v0, vcc
	buffer_store_dwordx2 v[2:3], v0, s[36:39], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v3_lds_padding
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
		.amdhsa_next_free_vgpr 420
		.amdhsa_next_free_sgpr 40
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
	.size	v3_lds_padding, .Lfunc_end0-v3_lds_padding
	.cfi_endproc
                                        ; -- End function
	.set v3_lds_padding.num_vgpr, 256
	.set v3_lds_padding.num_agpr, 164
	.set v3_lds_padding.numbered_sgpr, 40
	.set v3_lds_padding.num_named_barrier, 0
	.set v3_lds_padding.private_seg_size, 0
	.set v3_lds_padding.uses_vcc, 1
	.set v3_lds_padding.uses_flat_scratch, 0
	.set v3_lds_padding.has_dyn_sized_stack, 0
	.set v3_lds_padding.has_recursion, 0
	.set v3_lds_padding.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8496
; TotalNumSgprs: 46
; NumVgprs: 256
; NumAgprs: 164
; TotalNumVgprs: 420
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 52
; NumSGPRsForWavesPerEU: 46
; NumVGPRsForWavesPerEU: 420
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
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	103                             ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	159                             ; DW_AT_call_line
	.byte	33                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v3_lds" ; string offset=24
.Linfo_string3:
	.asciz	"v3_lds_padding"                ; string offset=78
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     164
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
    .name:           v3_lds_padding
    .private_segment_fixed_size: 0
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         v3_lds_padding.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     420
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
