	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v4_global_prefetch              ; -- Begin function v4_global_prefetch
	.p2align	8
	.type	v4_global_prefetch,@function
v4_global_prefetch:                     ; @v4_global_prefetch
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.8:
	.file	1 "/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v4_global_prefetch" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.9:
.LBB0_0:
	.file	2 "/root/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s9, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 8
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s17, 0, s1
	s_abs_i32 s15, s16
	v_readfirstlane_b32 s26, v0
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s14, s16, s0
	s_bfe_u32 s27, s26, 0x20006
	s_ashr_i32 s14, s14, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_lshlrev_b32_e32 v8, 3, v0
	v_and_b32_e32 v8, 56, v8
	s_mov_b32 s23, 0x27000
	v_readfirstlane_b32 s18, v1
	s_mul_i32 s17, s17, s18
	s_mul_hi_u32 s17, s18, s17
	s_add_i32 s18, s18, s17
	s_mul_hi_u32 s17, s15, s18
	s_mul_i32 s18, s17, s1
	s_sub_i32 s15, s15, s18
	s_add_i32 s18, s17, 1
	s_sub_i32 s19, s15, s1
	s_cmp_ge_u32 s15, s1
	s_cselect_b32 s17, s18, s17
	s_cselect_b32 s15, s19, s15
	s_add_i32 s18, s17, 1
	s_cmp_ge_u32 s15, s1
	s_cselect_b32 s1, s18, s17
	s_xor_b32 s1, s1, s14
	s_sub_i32 s1, s1, s14
	s_mul_i32 s0, s1, s0
	s_lshl_b32 s15, s1, 8
	s_sub_i32 s14, s16, s0
	s_mul_i32 s0, s15, s11
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s20, s2, s0
	v_lshlrev_b32_e32 v1, 1, v0
	s_addc_u32 s18, s3, s1
	s_lshl_b32 s14, s14, 8
	v_and_b32_e32 v1, 0x70, v1
	s_mul_i32 s16, s14, s12
	v_or_b32_e32 v17, s27, v1
	s_ashr_i32 s17, s16, 31
	v_or_b32_e32 v1, 4, v17
	v_or_b32_e32 v2, 8, v17
	v_or_b32_e32 v3, 12, v17
	v_or_b32_e32 v4, 0x80, v17
	v_or_b32_e32 v5, 0x84, v17
	v_or_b32_e32 v6, 0x88, v17
	v_or_b32_e32 v7, 0x8c, v17
	s_lshl_b64 s[24:25], s[16:17], 1
	s_add_u32 s16, s4, s24
	v_mul_lo_u32 v9, v17, s11
	v_mul_lo_u32 v10, v1, s11
	v_mul_lo_u32 v11, v2, s11
	v_mul_lo_u32 v12, v3, s11
	v_mul_lo_u32 v13, v4, s11
	v_mul_lo_u32 v14, v5, s11
	v_mul_lo_u32 v15, v6, s11
	v_mul_lo_u32 v16, v7, s11
	s_mul_i32 s11, s27, 0x420
	s_addc_u32 s17, s5, s25
	v_mul_lo_u32 v17, v17, s12
	v_mul_lo_u32 v1, v1, s12
	v_mul_lo_u32 v2, v2, s12
	v_mul_lo_u32 v3, v3, s12
	v_mul_lo_u32 v4, v4, s12
	v_mul_lo_u32 v5, v5, s12
	v_mul_lo_u32 v6, v6, s12
	v_mul_lo_u32 v7, v7, s12
	s_add_i32 s12, s11, 0
	s_and_b32 s21, s18, 0xffff
	s_mov_b32 s22, 0x7ffffffe
	v_add_lshl_u32 v9, v9, v8, 1
	s_mov_b32 m0, s12
	v_add_lshl_u32 v10, v10, v8, 1
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x1080
	v_add_lshl_u32 v11, v11, v8, 1
	buffer_load_dwordx4 v10, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x2100
	v_add_lshl_u32 v12, v12, v8, 1
	buffer_load_dwordx4 v11, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x3180
	v_add_lshl_u32 v13, v13, v8, 1
	buffer_load_dwordx4 v12, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x4200
	v_add_lshl_u32 v18, v14, v8, 1
	buffer_load_dwordx4 v13, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x5280
	v_add_lshl_u32 v15, v15, v8, 1
	buffer_load_dwordx4 v18, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x6300
	v_add_lshl_u32 v16, v16, v8, 1
	buffer_load_dwordx4 v15, s[20:23], 0 offen lds
	s_add_i32 m0, s12, 0x7380
	s_and_b32 s17, s17, 0xffff
	buffer_load_dwordx4 v16, s[20:23], 0 offen lds
	s_add_i32 s20, s12, 0x107e0
	s_mov_b32 s18, s22
	s_mov_b32 s19, s23
	v_add_lshl_u32 v17, v17, v8, 1
	s_mov_b32 m0, s20
	v_add_lshl_u32 v19, v1, v8, 1
	buffer_load_dwordx4 v17, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x11860
	v_add_lshl_u32 v2, v2, v8, 1
	buffer_load_dwordx4 v19, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x128e0
	v_add_lshl_u32 v255, v3, v8, 1
	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x13960
	v_add_lshl_u32 v254, v4, v8, 1
	buffer_load_dwordx4 v255, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x149e0
	v_add_lshl_u32 v4, v5, v8, 1
	buffer_load_dwordx4 v254, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x15a60
	v_add_lshl_u32 v5, v6, v8, 1
	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x16ae0
	v_add_lshl_u32 v6, v7, v8, 1
	buffer_load_dwordx4 v5, s[16:19], 0 offen lds
	s_add_i32 m0, s12, 0x17b60
	s_add_i32 s10, s10, 63
	buffer_load_dwordx4 v6, s[16:19], 0 offen lds
	s_ashr_i32 s11, s10, 31
	s_lshr_b32 s11, s11, 26
	s_add_i32 s11, s10, s11
	v_and_b32_e32 v1, 63, v0
	s_ashr_i32 s16, s11, 6
	v_lshl_or_b32 v1, s27, 6, v1
	s_cmpk_gt_i32 s10, 0x7f
	v_and_b32_e32 v3, 48, v0
	s_cbranch_scc1 .LBB0_3
; %bb.1:                                ; %.._crit_edge_crit_edge
	s_lshl_b32 s10, s26, 1
	s_and_b32 s17, s10, 0x80
	v_and_b32_e32 v14, 0xb0, v1
	v_accvgpr_write_b32 a0, s17
	s_add_i32 s10, s16, -1
	s_cbranch_execz .LBB0_4
; %bb.2:
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_read_b32 v7, a169
	v_accvgpr_read_b32 v13, a169
	v_accvgpr_read_b32 v45, a169
	v_accvgpr_read_b32 v6, a169
	v_accvgpr_read_b32 v5, a169
	v_accvgpr_read_b32 v4, a169
	v_accvgpr_write_b32 a137, v7
	v_accvgpr_read_b32 v12, a169
	v_accvgpr_read_b32 v11, a169
	v_accvgpr_read_b32 v10, a169
	v_accvgpr_write_b32 a145, v13
	v_accvgpr_read_b32 v44, a169
	v_accvgpr_read_b32 v43, a169
	v_accvgpr_read_b32 v42, a169
	v_accvgpr_write_b32 a141, v45
	v_accvgpr_write_b32 a128, v1
	v_accvgpr_mov_b32 a168, a169
	v_accvgpr_mov_b32 a167, a169
	v_accvgpr_mov_b32 a166, a169
	v_accvgpr_write_b32 a136, v6
	v_accvgpr_write_b32 a135, v5
	v_accvgpr_write_b32 a134, v4
	v_accvgpr_mov_b32 a177, a169
	v_accvgpr_mov_b32 a176, a169
	v_accvgpr_mov_b32 a175, a169
	v_accvgpr_mov_b32 a174, a169
	v_accvgpr_mov_b32 a161, a169
	v_accvgpr_mov_b32 a160, a169
	v_accvgpr_mov_b32 a159, a169
	v_accvgpr_mov_b32 a158, a169
	v_accvgpr_read_b32 v61, a169
	v_accvgpr_read_b32 v60, a169
	v_accvgpr_read_b32 v59, a169
	v_accvgpr_read_b32 v58, a169
	v_accvgpr_read_b32 v25, a169
	v_accvgpr_read_b32 v24, a169
	v_accvgpr_read_b32 v23, a169
	v_accvgpr_read_b32 v22, a169
	v_accvgpr_read_b32 v65, a169
	v_accvgpr_read_b32 v64, a169
	v_accvgpr_read_b32 v63, a169
	v_accvgpr_read_b32 v62, a169
	v_accvgpr_read_b32 v41, a169
	v_accvgpr_read_b32 v40, a169
	v_accvgpr_read_b32 v39, a169
	v_accvgpr_read_b32 v38, a169
	v_accvgpr_read_b32 v69, a169
	v_accvgpr_read_b32 v68, a169
	v_accvgpr_read_b32 v67, a169
	v_accvgpr_read_b32 v66, a169
	v_accvgpr_mov_b32 a181, a169
	v_accvgpr_mov_b32 a180, a169
	v_accvgpr_mov_b32 a179, a169
	v_accvgpr_mov_b32 a178, a169
	v_accvgpr_mov_b32 a185, a169
	v_accvgpr_mov_b32 a184, a169
	v_accvgpr_mov_b32 a183, a169
	v_accvgpr_mov_b32 a182, a169
	v_accvgpr_mov_b32 a189, a169
	v_accvgpr_mov_b32 a188, a169
	v_accvgpr_mov_b32 a187, a169
	v_accvgpr_mov_b32 a186, a169
	v_accvgpr_read_b32 v129, a169
	v_accvgpr_read_b32 v128, a169
	v_accvgpr_read_b32 v127, a169
	v_accvgpr_read_b32 v126, a169
	v_accvgpr_read_b32 v125, a169
	v_accvgpr_read_b32 v124, a169
	v_accvgpr_read_b32 v123, a169
	v_accvgpr_read_b32 v122, a169
	v_accvgpr_read_b32 v121, a169
	v_accvgpr_read_b32 v120, a169
	v_accvgpr_read_b32 v119, a169
	v_accvgpr_read_b32 v118, a169
	v_accvgpr_read_b32 v109, a169
	v_accvgpr_read_b32 v108, a169
	v_accvgpr_read_b32 v107, a169
	v_accvgpr_read_b32 v106, a169
	v_accvgpr_read_b32 v85, a169
	v_accvgpr_read_b32 v84, a169
	v_accvgpr_read_b32 v83, a169
	v_accvgpr_read_b32 v82, a169
	v_accvgpr_read_b32 v73, a169
	v_accvgpr_read_b32 v72, a169
	v_accvgpr_read_b32 v71, a169
	v_accvgpr_read_b32 v70, a169
	v_accvgpr_read_b32 v89, a169
	v_accvgpr_read_b32 v88, a169
	v_accvgpr_read_b32 v87, a169
	v_accvgpr_read_b32 v86, a169
	v_accvgpr_read_b32 v77, a169
	v_accvgpr_read_b32 v76, a169
	v_accvgpr_read_b32 v75, a169
	v_accvgpr_read_b32 v74, a169
	v_accvgpr_read_b32 v97, a169
	v_accvgpr_read_b32 v96, a169
	v_accvgpr_read_b32 v95, a169
	v_accvgpr_read_b32 v94, a169
	v_accvgpr_read_b32 v81, a169
	v_accvgpr_read_b32 v80, a169
	v_accvgpr_read_b32 v79, a169
	v_accvgpr_read_b32 v78, a169
	v_accvgpr_read_b32 v117, a169
	v_accvgpr_read_b32 v116, a169
	v_accvgpr_read_b32 v115, a169
	v_accvgpr_read_b32 v114, a169
	v_accvgpr_read_b32 v93, a169
	v_accvgpr_read_b32 v92, a169
	v_accvgpr_read_b32 v91, a169
	v_accvgpr_read_b32 v90, a169
	v_accvgpr_read_b32 v105, a169
	v_accvgpr_read_b32 v104, a169
	v_accvgpr_read_b32 v103, a169
	v_accvgpr_read_b32 v102, a169
	v_accvgpr_read_b32 v113, a169
	v_accvgpr_read_b32 v112, a169
	v_accvgpr_read_b32 v111, a169
	v_accvgpr_read_b32 v110, a169
	v_accvgpr_read_b32 v133, a169
	v_accvgpr_read_b32 v132, a169
	v_accvgpr_read_b32 v131, a169
	v_accvgpr_read_b32 v130, a169
	v_accvgpr_read_b32 v149, a169
	v_accvgpr_read_b32 v148, a169
	v_accvgpr_read_b32 v147, a169
	v_accvgpr_read_b32 v146, a169
	v_accvgpr_read_b32 v161, a169
	v_accvgpr_read_b32 v160, a169
	v_accvgpr_read_b32 v159, a169
	v_accvgpr_read_b32 v158, a169
	v_accvgpr_read_b32 v169, a169
	v_accvgpr_read_b32 v168, a169
	v_accvgpr_read_b32 v167, a169
	v_accvgpr_read_b32 v166, a169
	v_accvgpr_read_b32 v173, a169
	v_accvgpr_read_b32 v172, a169
	v_accvgpr_read_b32 v171, a169
	v_accvgpr_read_b32 v170, a169
	v_accvgpr_read_b32 v153, a169
	v_accvgpr_read_b32 v152, a169
	v_accvgpr_read_b32 v151, a169
	v_accvgpr_read_b32 v150, a169
	v_accvgpr_read_b32 v101, a169
	v_accvgpr_read_b32 v100, a169
	v_accvgpr_read_b32 v99, a169
	v_accvgpr_read_b32 v98, a169
	v_accvgpr_read_b32 v177, a169
	v_accvgpr_read_b32 v176, a169
	v_accvgpr_read_b32 v175, a169
	v_accvgpr_read_b32 v174, a169
	v_accvgpr_read_b32 v165, a169
	v_accvgpr_read_b32 v164, a169
	v_accvgpr_read_b32 v163, a169
	v_accvgpr_read_b32 v162, a169
	v_accvgpr_read_b32 v189, a169
	v_accvgpr_read_b32 v188, a169
	v_accvgpr_read_b32 v187, a169
	v_accvgpr_read_b32 v186, a169
	v_accvgpr_read_b32 v157, a169
	v_accvgpr_read_b32 v156, a169
	v_accvgpr_read_b32 v155, a169
	v_accvgpr_read_b32 v154, a169
	v_accvgpr_read_b32 v213, a169
	v_accvgpr_read_b32 v212, a169
	v_accvgpr_read_b32 v211, a169
	v_accvgpr_read_b32 v210, a169
	v_accvgpr_read_b32 v185, a169
	v_accvgpr_read_b32 v184, a169
	v_accvgpr_read_b32 v183, a169
	v_accvgpr_read_b32 v182, a169
	v_accvgpr_read_b32 v221, a169
	v_accvgpr_read_b32 v220, a169
	v_accvgpr_read_b32 v219, a169
	v_accvgpr_read_b32 v218, a169
	v_accvgpr_read_b32 v205, a169
	v_accvgpr_read_b32 v204, a169
	v_accvgpr_read_b32 v203, a169
	v_accvgpr_read_b32 v202, a169
	v_accvgpr_read_b32 v197, a169
	v_accvgpr_read_b32 v196, a169
	v_accvgpr_read_b32 v195, a169
	v_accvgpr_read_b32 v194, a169
	v_accvgpr_read_b32 v181, a169
	v_accvgpr_read_b32 v180, a169
	v_accvgpr_read_b32 v179, a169
	v_accvgpr_read_b32 v178, a169
	v_accvgpr_read_b32 v209, a169
	v_accvgpr_read_b32 v208, a169
	v_accvgpr_read_b32 v207, a169
	v_accvgpr_read_b32 v206, a169
	v_accvgpr_read_b32 v201, a169
	v_accvgpr_read_b32 v200, a169
	v_accvgpr_read_b32 v199, a169
	v_accvgpr_read_b32 v198, a169
	v_accvgpr_read_b32 v217, a169
	v_accvgpr_read_b32 v216, a169
	v_accvgpr_read_b32 v215, a169
	v_accvgpr_read_b32 v214, a169
	v_accvgpr_read_b32 v193, a169
	v_accvgpr_read_b32 v192, a169
	v_accvgpr_read_b32 v191, a169
	v_accvgpr_read_b32 v190, a169
	v_accvgpr_read_b32 v245, a169
	v_accvgpr_read_b32 v244, a169
	v_accvgpr_read_b32 v243, a169
	v_accvgpr_read_b32 v242, a169
	v_accvgpr_read_b32 v225, a169
	v_accvgpr_read_b32 v224, a169
	v_accvgpr_read_b32 v223, a169
	v_accvgpr_read_b32 v222, a169
	v_accvgpr_read_b32 v237, a169
	v_accvgpr_read_b32 v236, a169
	v_accvgpr_read_b32 v235, a169
	v_accvgpr_read_b32 v234, a169
	v_accvgpr_read_b32 v233, a169
	v_accvgpr_read_b32 v232, a169
	v_accvgpr_read_b32 v231, a169
	v_accvgpr_read_b32 v230, a169
	v_accvgpr_read_b32 v241, a169
	v_accvgpr_read_b32 v240, a169
	v_accvgpr_read_b32 v239, a169
	v_accvgpr_read_b32 v238, a169
	v_accvgpr_read_b32 v229, a169
	v_accvgpr_read_b32 v228, a169
	v_accvgpr_read_b32 v227, a169
	v_accvgpr_read_b32 v226, a169
	v_accvgpr_read_b32 v9, a169
	v_accvgpr_read_b32 v8, a169
	v_accvgpr_write_b32 a144, v12
	v_accvgpr_write_b32 a143, v11
	v_accvgpr_write_b32 a142, v10
	v_accvgpr_read_b32 v49, a169
	v_accvgpr_read_b32 v48, a169
	v_accvgpr_read_b32 v47, a169
	v_accvgpr_read_b32 v46, a169
	v_accvgpr_read_b32 v57, a169
	v_accvgpr_read_b32 v56, a169
	v_accvgpr_read_b32 v55, a169
	v_accvgpr_read_b32 v54, a169
	v_accvgpr_read_b32 v29, a169
	v_accvgpr_read_b32 v28, a169
	v_accvgpr_read_b32 v27, a169
	v_accvgpr_read_b32 v26, a169
	v_accvgpr_read_b32 v33, a169
	v_accvgpr_read_b32 v32, a169
	v_accvgpr_read_b32 v31, a169
	v_accvgpr_read_b32 v30, a169
	v_accvgpr_read_b32 v37, a169
	v_accvgpr_read_b32 v36, a169
	v_accvgpr_read_b32 v35, a169
	v_accvgpr_read_b32 v34, a169
	v_accvgpr_read_b32 v249, a169
	v_accvgpr_read_b32 v248, a169
	v_accvgpr_read_b32 v247, a169
	v_accvgpr_read_b32 v246, a169
	v_accvgpr_write_b32 a140, v44
	v_accvgpr_write_b32 a139, v43
	v_accvgpr_write_b32 a138, v42
	v_accvgpr_read_b32 v1, a0
	s_branch .LBB0_7
.LBB0_3:
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $agpr0
	s_add_i32 s10, s16, -1
.LBB0_4:                                ; %.lr.ph
	v_accvgpr_write_b32 a124, v0
	v_lshlrev_b32_e32 v0, 10, v0
	s_lshl_b32 s16, s26, 1
	v_and_b32_e32 v0, 0x3c00, v0
	s_and_b32 s16, s16, 0x80
	v_accvgpr_write_b32 a128, v1
	v_accvgpr_write_b32 a131, v6
	v_and_b32_e32 v6, 0xb0, v1
	v_lshrrev_b32_e32 v1, 5, v0
	v_accvgpr_write_b32 a132, v2
	v_add_u32_e32 v2, 0, v0
	s_add_i32 s17, s16, 0
	v_add3_u32 v2, v2, v6, v1
	s_add_i32 s17, s17, 0x107e0
	v_accvgpr_write_b32 a133, v2
	v_add_u32_e32 v2, s17, v3
	s_max_i32 s17, s10, 1
	s_add_u32 s4, s4, s24
	s_addc_u32 s5, s5, s25
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	s_add_u32 s0, s2, s0
	v_mov_b32_e32 v42, 0
	v_accvgpr_write_b32 a125, v3
	v_add3_u32 v0, v2, v0, v1
	s_addc_u32 s1, s3, s1
	v_mov_b32_e32 v3, v42
	v_accvgpr_write_b32 a134, v0
	v_mov_b32_e32 v0, v42
	s_add_u32 s18, s0, 0x80
	s_mov_b32 s11, 0
	v_mov_b32_e32 v1, v42
	v_mov_b32_e32 v2, v42
	v_accvgpr_write_b32 a149, v3
	v_accvgpr_write_b32 a126, v6
	s_addc_u32 s19, s1, 0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_mov_b32_e32 v43, v42
	v_accvgpr_write_b32 a153, v3
	v_mov_b32_e32 v44, v42
	v_mov_b32_e32 v45, v42
	v_accvgpr_write_b32 a154, v42
	v_accvgpr_write_b32 a155, v42
	v_accvgpr_write_b32 a156, v42
	v_accvgpr_write_b32 a157, v42
	v_accvgpr_write_b32 a152, v2
	v_accvgpr_write_b32 a151, v1
	v_accvgpr_write_b32 a150, v0
	v_accvgpr_write_b32 a135, v10
	v_accvgpr_write_b32 a136, v11
	v_accvgpr_write_b32 a137, v12
	v_accvgpr_write_b32 a138, v13
	v_accvgpr_write_b32 a148, v2
	v_accvgpr_write_b32 a147, v1
	v_accvgpr_write_b32 a146, v0
	v_mov_b32_e32 v34, v42
	v_mov_b32_e32 v35, v42
	v_mov_b32_e32 v36, v42
	v_mov_b32_e32 v37, v42
	v_mov_b32_e32 v30, v42
	v_mov_b32_e32 v31, v42
	v_mov_b32_e32 v32, v42
	v_mov_b32_e32 v33, v42
	v_mov_b32_e32 v26, v42
	v_mov_b32_e32 v27, v42
	v_mov_b32_e32 v28, v42
	v_mov_b32_e32 v29, v42
	v_mov_b32_e32 v54, v42
	v_mov_b32_e32 v55, v42
	v_mov_b32_e32 v56, v42
	v_mov_b32_e32 v57, v42
	v_mov_b32_e32 v46, v42
	v_mov_b32_e32 v47, v42
	v_mov_b32_e32 v48, v42
	v_mov_b32_e32 v49, v42
	v_accvgpr_write_b32 a170, v42
	v_accvgpr_write_b32 a171, v42
	v_accvgpr_write_b32 a172, v42
	v_accvgpr_write_b32 a139, v9
	v_accvgpr_write_b32 a173, v42
	v_mov_b32_e32 v6, v42
	v_mov_b32_e32 v7, v42
	v_mov_b32_e32 v8, v42
	v_mov_b32_e32 v9, v42
	v_mov_b32_e32 v226, v42
	v_mov_b32_e32 v227, v42
	v_mov_b32_e32 v228, v42
	v_mov_b32_e32 v229, v42
	v_mov_b32_e32 v238, v42
	v_mov_b32_e32 v239, v42
	v_mov_b32_e32 v240, v42
	v_mov_b32_e32 v241, v42
	v_mov_b32_e32 v230, v42
	v_mov_b32_e32 v231, v42
	v_mov_b32_e32 v232, v42
	v_mov_b32_e32 v233, v42
	v_mov_b32_e32 v234, v42
	v_mov_b32_e32 v235, v42
	v_mov_b32_e32 v236, v42
	v_mov_b32_e32 v237, v42
	v_mov_b32_e32 v222, v42
	v_mov_b32_e32 v223, v42
	v_mov_b32_e32 v224, v42
	v_mov_b32_e32 v225, v42
	v_mov_b32_e32 v242, v42
	v_mov_b32_e32 v243, v42
	v_mov_b32_e32 v244, v42
	v_mov_b32_e32 v245, v42
	v_mov_b32_e32 v190, v42
	v_mov_b32_e32 v191, v42
	v_mov_b32_e32 v192, v42
	v_mov_b32_e32 v193, v42
	v_mov_b32_e32 v214, v42
	v_mov_b32_e32 v215, v42
	v_mov_b32_e32 v216, v42
	v_mov_b32_e32 v217, v42
	v_mov_b32_e32 v198, v42
	v_mov_b32_e32 v199, v42
	v_mov_b32_e32 v200, v42
	v_mov_b32_e32 v201, v42
	v_mov_b32_e32 v206, v42
	v_mov_b32_e32 v207, v42
	v_mov_b32_e32 v208, v42
	v_mov_b32_e32 v209, v42
	v_mov_b32_e32 v178, v42
	v_mov_b32_e32 v179, v42
	v_mov_b32_e32 v180, v42
	v_mov_b32_e32 v181, v42
	v_mov_b32_e32 v194, v42
	v_mov_b32_e32 v195, v42
	v_mov_b32_e32 v196, v42
	v_mov_b32_e32 v197, v42
	v_mov_b32_e32 v202, v42
	v_mov_b32_e32 v203, v42
	v_mov_b32_e32 v204, v42
	v_mov_b32_e32 v205, v42
	v_mov_b32_e32 v218, v42
	v_mov_b32_e32 v219, v42
	v_mov_b32_e32 v220, v42
	v_mov_b32_e32 v221, v42
	v_mov_b32_e32 v182, v42
	v_mov_b32_e32 v183, v42
	v_mov_b32_e32 v184, v42
	v_mov_b32_e32 v185, v42
	v_mov_b32_e32 v210, v42
	v_mov_b32_e32 v211, v42
	v_mov_b32_e32 v212, v42
	v_mov_b32_e32 v213, v42
	v_mov_b32_e32 v154, v42
	v_mov_b32_e32 v155, v42
	v_mov_b32_e32 v156, v42
	v_mov_b32_e32 v157, v42
	v_mov_b32_e32 v186, v42
	v_mov_b32_e32 v187, v42
	v_mov_b32_e32 v188, v42
	v_mov_b32_e32 v189, v42
	v_mov_b32_e32 v162, v42
	v_mov_b32_e32 v163, v42
	v_mov_b32_e32 v164, v42
	v_mov_b32_e32 v165, v42
	v_mov_b32_e32 v174, v42
	v_mov_b32_e32 v175, v42
	v_mov_b32_e32 v176, v42
	v_mov_b32_e32 v177, v42
	v_mov_b32_e32 v98, v42
	v_mov_b32_e32 v99, v42
	v_mov_b32_e32 v100, v42
	v_mov_b32_e32 v101, v42
	v_mov_b32_e32 v150, v42
	v_mov_b32_e32 v151, v42
	v_mov_b32_e32 v152, v42
	v_mov_b32_e32 v153, v42
	v_mov_b32_e32 v170, v42
	v_mov_b32_e32 v171, v42
	v_mov_b32_e32 v172, v42
	v_mov_b32_e32 v173, v42
	v_mov_b32_e32 v166, v42
	v_mov_b32_e32 v167, v42
	v_mov_b32_e32 v168, v42
	v_mov_b32_e32 v169, v42
	v_mov_b32_e32 v158, v42
	v_mov_b32_e32 v159, v42
	v_mov_b32_e32 v160, v42
	v_mov_b32_e32 v161, v42
	v_mov_b32_e32 v146, v42
	v_mov_b32_e32 v147, v42
	v_mov_b32_e32 v148, v42
	v_mov_b32_e32 v149, v42
	v_mov_b32_e32 v130, v42
	v_mov_b32_e32 v131, v42
	v_mov_b32_e32 v132, v42
	v_mov_b32_e32 v133, v42
	v_mov_b32_e32 v110, v42
	v_mov_b32_e32 v111, v42
	v_mov_b32_e32 v112, v42
	v_mov_b32_e32 v113, v42
	v_mov_b32_e32 v102, v42
	v_mov_b32_e32 v103, v42
	v_mov_b32_e32 v104, v42
	v_mov_b32_e32 v105, v42
	v_mov_b32_e32 v90, v42
	v_mov_b32_e32 v91, v42
	v_mov_b32_e32 v92, v42
	v_mov_b32_e32 v93, v42
	v_mov_b32_e32 v114, v42
	v_mov_b32_e32 v115, v42
	v_mov_b32_e32 v116, v42
	v_mov_b32_e32 v117, v42
	v_mov_b32_e32 v78, v42
	v_mov_b32_e32 v79, v42
	v_mov_b32_e32 v80, v42
	v_mov_b32_e32 v81, v42
	v_mov_b32_e32 v94, v42
	v_mov_b32_e32 v95, v42
	v_mov_b32_e32 v96, v42
	v_mov_b32_e32 v97, v42
	v_mov_b32_e32 v74, v42
	v_mov_b32_e32 v75, v42
	v_mov_b32_e32 v76, v42
	v_mov_b32_e32 v77, v42
	v_mov_b32_e32 v86, v42
	v_mov_b32_e32 v87, v42
	v_mov_b32_e32 v88, v42
	v_mov_b32_e32 v89, v42
	v_mov_b32_e32 v70, v42
	v_mov_b32_e32 v71, v42
	v_mov_b32_e32 v72, v42
	v_mov_b32_e32 v73, v42
	v_mov_b32_e32 v82, v42
	v_mov_b32_e32 v83, v42
	v_mov_b32_e32 v84, v42
	v_mov_b32_e32 v85, v42
	v_mov_b32_e32 v106, v42
	v_mov_b32_e32 v107, v42
	v_mov_b32_e32 v108, v42
	v_mov_b32_e32 v109, v42
	v_mov_b32_e32 v118, v42
	v_mov_b32_e32 v119, v42
	v_mov_b32_e32 v120, v42
	v_mov_b32_e32 v121, v42
	v_mov_b32_e32 v122, v42
	v_mov_b32_e32 v123, v42
	v_mov_b32_e32 v124, v42
	v_mov_b32_e32 v125, v42
	v_mov_b32_e32 v126, v42
	v_mov_b32_e32 v127, v42
	v_mov_b32_e32 v128, v42
	v_mov_b32_e32 v129, v42
	v_accvgpr_write_b32 a186, v42
	v_accvgpr_write_b32 a187, v42
	v_accvgpr_write_b32 a188, v42
	v_accvgpr_write_b32 a189, v42
	v_accvgpr_write_b32 a182, v42
	v_accvgpr_write_b32 a183, v42
	v_accvgpr_write_b32 a184, v42
	v_accvgpr_write_b32 a185, v42
	v_accvgpr_write_b32 a178, v42
	v_accvgpr_write_b32 a179, v42
	v_accvgpr_write_b32 a180, v42
	v_accvgpr_write_b32 a181, v42
	v_mov_b32_e32 v66, v42
	v_mov_b32_e32 v67, v42
	v_mov_b32_e32 v68, v42
	v_mov_b32_e32 v69, v42
	v_mov_b32_e32 v38, v42
	v_mov_b32_e32 v39, v42
	v_mov_b32_e32 v40, v42
	v_mov_b32_e32 v41, v42
	v_mov_b32_e32 v62, v42
	v_mov_b32_e32 v63, v42
	v_mov_b32_e32 v64, v42
	v_mov_b32_e32 v65, v42
	v_mov_b32_e32 v22, v42
	v_mov_b32_e32 v23, v42
	v_mov_b32_e32 v24, v42
	v_mov_b32_e32 v25, v42
	v_mov_b32_e32 v58, v42
	v_mov_b32_e32 v59, v42
	v_mov_b32_e32 v60, v42
	v_mov_b32_e32 v61, v42
	v_accvgpr_write_b32 a158, v42
	v_accvgpr_write_b32 a159, v42
	v_accvgpr_write_b32 a160, v42
	v_accvgpr_write_b32 a161, v42
	v_accvgpr_write_b32 a140, v254
	v_accvgpr_write_b32 a141, v18
	v_accvgpr_write_b32 a174, v42
	v_mov_b32_e32 v254, v5
	v_mov_b32_e32 v5, v19
	v_accvgpr_write_b32 a175, v42
	v_accvgpr_write_b32 a176, v42
	v_accvgpr_write_b32 a177, v42
	v_accvgpr_write_b32 a8, v42
	v_accvgpr_write_b32 a9, v42
	v_accvgpr_write_b32 a10, v42
	v_accvgpr_write_b32 a11, v42
	v_accvgpr_write_b32 a166, v42
	v_accvgpr_write_b32 a167, v42
	v_accvgpr_write_b32 a168, v42
	v_accvgpr_write_b32 a169, v42
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s21, s11, 1
	s_lshl_b32 s0, s21, 14
	s_xor_b32 s0, s0, 0x4000
	s_lshr_b32 s1, s0, 5
	s_or_b32 s0, s1, s0
	s_lshl_b32 s22, s0, 1
	s_add_i32 s23, s12, s22
	s_mov_b32 m0, s23
	s_and_b32 s1, s19, 0xffff
	s_mov_b32 s0, s18
	v_accvgpr_read_b32 v0, a139
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	s_add_i32 m0, s23, 0x1080
	v_accvgpr_read_b32 v0, a135
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	s_add_i32 m0, s23, 0x2100
	v_accvgpr_read_b32 v0, a136
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	s_add_i32 m0, s23, 0x3180
	v_accvgpr_read_b32 v0, a137
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	s_add_i32 m0, s23, 0x4200
	v_accvgpr_read_b32 v0, a138
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	s_add_i32 m0, s23, 0x5280
	v_accvgpr_read_b32 v0, a141
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	s_add_i32 m0, s23, 0x6300
	s_add_i32 s22, s20, s22
	buffer_load_dwordx4 v15, s[0:3], 0 offen lds
	s_add_i32 m0, s23, 0x7380
	v_accvgpr_read_b32 v0, a132
	buffer_load_dwordx4 v16, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_and_b32 s1, s5, 0xffff
	s_mov_b32 s0, s4
	buffer_load_dwordx4 v17, s[0:3], 0 offen lds
	s_add_i32 m0, s22, 0x1080
	s_mul_i32 s21, s21, 0x8400
	buffer_load_dwordx4 v5, s[0:3], 0 offen lds
	s_add_i32 m0, s22, 0x2100
	v_accvgpr_mov_b32 a145, a11
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	s_add_i32 m0, s22, 0x3180
	v_accvgpr_read_b32 v0, a140
	buffer_load_dwordx4 v255, s[0:3], 0 offen lds
	s_add_i32 m0, s22, 0x4200
	v_accvgpr_mov_b32 a144, a10
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	s_add_i32 m0, s22, 0x5280
	v_accvgpr_read_b32 v0, a131
	buffer_load_dwordx4 v4, s[0:3], 0 offen lds
	s_add_i32 m0, s22, 0x6300
	v_accvgpr_mov_b32 a143, a9
	buffer_load_dwordx4 v254, s[0:3], 0 offen lds
	s_add_i32 m0, s22, 0x7380
	v_accvgpr_mov_b32 a142, a8
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	v_accvgpr_read_b32 v0, a134
	v_add_u32_e32 v0, s21, v0
	s_waitcnt vmcnt(16) lgkmcnt(0)
	s_barrier
	ds_read_b128 a[96:99], v0
	ds_read_b128 a[64:67], v0 offset:256
	ds_read_b128 a[68:71], v0 offset:512
	ds_read_b128 a[72:75], v0 offset:768
	ds_read_b128 a[76:79], v0 offset:16896
	ds_read_b128 a[80:83], v0 offset:17152
	ds_read_b128 a[84:87], v0 offset:17408
	ds_read_b128 a[88:91], v0 offset:17664
	ds_read_b128 a[0:3], v0 offset:17728
	ds_read_b128 a[8:11], v0 offset:64
	ds_read_b128 a[12:15], v0 offset:320
	ds_read_b128 a[16:19], v0 offset:576
	ds_read_b128 a[20:23], v0 offset:832
	ds_read_b128 a[24:27], v0 offset:16960
	ds_read_b128 a[28:31], v0 offset:17216
	ds_read_b128 a[32:35], v0 offset:17472
	v_accvgpr_read_b32 v0, a133
	v_add_u32_e32 v0, s21, v0
	ds_read_b128 a[100:103], v0
	ds_read_b128 a[104:107], v0 offset:256
	v_accvgpr_write_b32 a127, v15
	v_accvgpr_write_b32 a129, v16
	v_accvgpr_write_b32 a130, v17
	ds_read_b128 v[14:17], v0 offset:512
	ds_read_b128 a[108:111], v0 offset:768
	ds_read_b128 a[112:115], v0 offset:16896
	ds_read_b128 a[116:119], v0 offset:17152
	ds_read_b128 a[120:123], v0 offset:17408
	ds_read_b128 a[92:95], v0 offset:17664
	ds_read_b128 a[4:7], v0 offset:17728
	ds_read_b128 a[60:63], v0 offset:64
	ds_read_b128 a[56:59], v0 offset:320
	ds_read_b128 a[52:55], v0 offset:576
	ds_read_b128 a[48:51], v0 offset:832
	ds_read_b128 a[44:47], v0 offset:16960
	ds_read_b128 a[40:43], v0 offset:17216
	ds_read_b128 a[36:39], v0 offset:17472
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 a[154:157], a[64:67], a[100:103], a[154:157]
	v_accvgpr_mov_b32 a165, a161
	v_accvgpr_mov_b32 a164, a160
	v_accvgpr_mov_b32 a163, a159
	v_mfma_f32_16x16x32_f16 a[150:153], a[68:71], a[100:103], a[150:153]
	v_accvgpr_mov_b32 a162, a158
	s_add_i32 s11, s11, 1
	s_add_u32 s4, s4, 0x80
	v_mfma_f32_16x16x32_f16 v[6:9], a[68:71], a[104:107], v[6:9]
	s_addc_u32 s5, s5, 0
	s_add_u32 s18, s18, 0x80
	s_addc_u32 s19, s19, 0
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[214:217], a[68:71], v[14:17], v[214:217]
	s_cmp_lg_u32 s17, s11
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x32_f16 v[210:213], a[68:71], a[108:111], v[210:213]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[166:169], a[68:71], a[112:115], v[166:169]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_f16 v[78:81], a[68:71], a[116:119], v[78:81]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_f16 v[122:125], a[68:71], a[120:123], v[122:125]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_f16 v[22:25], a[68:71], a[92:95], v[22:25]
	v_accvgpr_mov_b32 a68, a142
	v_accvgpr_mov_b32 a69, a143
	v_accvgpr_mov_b32 a70, a144
	v_accvgpr_mov_b32 a71, a145
	v_mfma_f32_16x16x32_f16 v[226:229], a[72:75], a[104:107], v[226:229]
	v_mfma_f32_16x16x32_f16 v[238:241], a[76:79], a[104:107], v[238:241]
	v_mfma_f32_16x16x32_f16 v[230:233], a[80:83], a[104:107], v[230:233]
	v_mfma_f32_16x16x32_f16 v[234:237], a[84:87], a[104:107], v[234:237]
	v_mfma_f32_16x16x32_f16 v[222:225], a[88:91], a[104:107], v[222:225]
	v_mfma_f32_16x16x32_f16 v[242:245], a[96:99], v[14:17], v[242:245]
	v_mfma_f32_16x16x32_f16 v[190:193], a[64:67], v[14:17], v[190:193]
	v_mfma_f32_16x16x32_f16 v[198:201], a[72:75], v[14:17], v[198:201]
	v_mfma_f32_16x16x32_f16 v[206:209], a[76:79], v[14:17], v[206:209]
	v_mfma_f32_16x16x32_f16 v[178:181], a[80:83], v[14:17], v[178:181]
	v_mfma_f32_16x16x32_f16 v[194:197], a[84:87], v[14:17], v[194:197]
	v_mfma_f32_16x16x32_f16 v[202:205], a[88:91], v[14:17], v[202:205]
	v_accvgpr_read_b32 v17, a130
	v_accvgpr_read_b32 v16, a129
	v_accvgpr_read_b32 v15, a127
	v_mfma_f32_16x16x32_f16 v[218:221], a[96:99], a[108:111], v[218:221]
	v_mfma_f32_16x16x32_f16 v[182:185], a[64:67], a[108:111], v[182:185]
	v_mfma_f32_16x16x32_f16 v[154:157], a[72:75], a[108:111], v[154:157]
	v_mfma_f32_16x16x32_f16 v[186:189], a[76:79], a[108:111], v[186:189]
	v_mfma_f32_16x16x32_f16 v[162:165], a[80:83], a[108:111], v[162:165]
	v_mfma_f32_16x16x32_f16 v[174:177], a[84:87], a[108:111], v[174:177]
	v_mfma_f32_16x16x32_f16 v[98:101], a[88:91], a[108:111], v[98:101]
	v_mfma_f32_16x16x32_f16 v[150:153], a[96:99], a[112:115], v[150:153]
	v_mfma_f32_16x16x32_f16 v[170:173], a[64:67], a[112:115], v[170:173]
	v_mfma_f32_16x16x32_f16 v[158:161], a[72:75], a[112:115], v[158:161]
	v_mfma_f32_16x16x32_f16 v[146:149], a[76:79], a[112:115], v[146:149]
	v_mfma_f32_16x16x32_f16 v[130:133], a[80:83], a[112:115], v[130:133]
	v_mfma_f32_16x16x32_f16 v[110:113], a[84:87], a[112:115], v[110:113]
	v_mfma_f32_16x16x32_f16 v[102:105], a[88:91], a[112:115], v[102:105]
	v_mfma_f32_16x16x32_f16 v[90:93], a[96:99], a[116:119], v[90:93]
	v_mfma_f32_16x16x32_f16 v[114:117], a[64:67], a[116:119], v[114:117]
	v_mfma_f32_16x16x32_f16 v[94:97], a[72:75], a[116:119], v[94:97]
	v_mfma_f32_16x16x32_f16 v[74:77], a[76:79], a[116:119], v[74:77]
	v_mfma_f32_16x16x32_f16 v[86:89], a[80:83], a[116:119], v[86:89]
	v_mfma_f32_16x16x32_f16 v[70:73], a[84:87], a[116:119], v[70:73]
	v_mfma_f32_16x16x32_f16 v[82:85], a[88:91], a[116:119], v[82:85]
	v_mfma_f32_16x16x32_f16 v[106:109], a[96:99], a[120:123], v[106:109]
	v_mfma_f32_16x16x32_f16 v[118:121], a[64:67], a[120:123], v[118:121]
	v_mfma_f32_16x16x32_f16 v[126:129], a[72:75], a[120:123], v[126:129]
	v_mfma_f32_16x16x32_f16 a[186:189], a[76:79], a[120:123], a[186:189]
	v_mfma_f32_16x16x32_f16 a[116:119], a[80:83], a[120:123], a[182:185]
	v_mfma_f32_16x16x32_f16 a[112:115], a[84:87], a[120:123], a[178:181]
	v_mfma_f32_16x16x32_f16 v[66:69], a[88:91], a[120:123], v[66:69]
	v_mfma_f32_16x16x32_f16 a[108:111], a[64:67], a[104:107], a[170:173]
	v_mfma_f32_16x16x32_f16 v[46:49], a[96:99], a[104:107], v[46:49]
	v_mfma_f32_16x16x32_f16 v[54:57], a[88:91], a[100:103], v[54:57]
	v_mfma_f32_16x16x32_f16 v[26:29], a[84:87], a[100:103], v[26:29]
	v_mfma_f32_16x16x32_f16 v[30:33], a[80:83], a[100:103], v[30:33]
	v_mfma_f32_16x16x32_f16 v[34:37], a[76:79], a[100:103], v[34:37]
	v_mfma_f32_16x16x32_f16 a[146:149], a[72:75], a[100:103], a[146:149]
	v_mfma_f32_16x16x32_f16 v[42:45], a[96:99], a[100:103], v[42:45]
	v_mfma_f32_16x16x32_f16 v[38:41], a[96:99], a[92:95], v[38:41]
	v_mfma_f32_16x16x32_f16 v[62:65], a[64:67], a[92:95], v[62:65]
	v_mfma_f32_16x16x32_f16 v[58:61], a[72:75], a[92:95], v[58:61]
	v_mfma_f32_16x16x32_f16 a[64:67], a[76:79], a[92:95], a[162:165]
	v_accvgpr_mov_b32 a76, a154
	v_accvgpr_mov_b32 a77, a155
	v_accvgpr_mov_b32 a78, a156
	v_mfma_f32_16x16x32_f16 a[80:83], a[80:83], a[92:95], a[174:177]
	v_accvgpr_mov_b32 a79, a157
	v_mfma_f32_16x16x32_f16 a[72:75], a[84:87], a[92:95], a[68:71]
	v_mfma_f32_16x16x32_f16 a[68:71], a[88:91], a[92:95], a[166:169]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_f16 v[42:45], a[8:11], a[60:63], v[42:45]
	v_mfma_f32_16x16x32_f16 a[154:157], a[12:15], a[60:63], a[76:79]
	v_mfma_f32_16x16x32_f16 a[150:153], a[16:19], a[60:63], a[150:153]
	v_mfma_f32_16x16x32_f16 a[146:149], a[20:23], a[60:63], a[146:149]
	v_mfma_f32_16x16x32_f16 v[34:37], a[24:27], a[60:63], v[34:37]
	v_mfma_f32_16x16x32_f16 v[30:33], a[28:31], a[60:63], v[30:33]
	v_mfma_f32_16x16x32_f16 v[26:29], a[32:35], a[60:63], v[26:29]
	v_mfma_f32_16x16x32_f16 v[54:57], a[0:3], a[60:63], v[54:57]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[46:49], a[8:11], a[56:59], v[46:49]
	v_mfma_f32_16x16x32_f16 a[170:173], a[12:15], a[56:59], a[108:111]
	v_mfma_f32_16x16x32_f16 v[6:9], a[16:19], a[56:59], v[6:9]
	v_mfma_f32_16x16x32_f16 v[226:229], a[20:23], a[56:59], v[226:229]
	v_mfma_f32_16x16x32_f16 v[238:241], a[24:27], a[56:59], v[238:241]
	v_mfma_f32_16x16x32_f16 v[230:233], a[28:31], a[56:59], v[230:233]
	v_mfma_f32_16x16x32_f16 v[234:237], a[32:35], a[56:59], v[234:237]
	v_mfma_f32_16x16x32_f16 v[222:225], a[0:3], a[56:59], v[222:225]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 v[242:245], a[8:11], a[52:55], v[242:245]
	v_mfma_f32_16x16x32_f16 v[190:193], a[12:15], a[52:55], v[190:193]
	v_mfma_f32_16x16x32_f16 v[214:217], a[16:19], a[52:55], v[214:217]
	v_mfma_f32_16x16x32_f16 v[198:201], a[20:23], a[52:55], v[198:201]
	v_mfma_f32_16x16x32_f16 v[206:209], a[24:27], a[52:55], v[206:209]
	v_mfma_f32_16x16x32_f16 v[178:181], a[28:31], a[52:55], v[178:181]
	v_mfma_f32_16x16x32_f16 v[194:197], a[32:35], a[52:55], v[194:197]
	v_mfma_f32_16x16x32_f16 v[202:205], a[0:3], a[52:55], v[202:205]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[218:221], a[8:11], a[48:51], v[218:221]
	v_mfma_f32_16x16x32_f16 v[182:185], a[12:15], a[48:51], v[182:185]
	v_mfma_f32_16x16x32_f16 v[210:213], a[16:19], a[48:51], v[210:213]
	v_mfma_f32_16x16x32_f16 v[154:157], a[20:23], a[48:51], v[154:157]
	v_mfma_f32_16x16x32_f16 v[186:189], a[24:27], a[48:51], v[186:189]
	v_mfma_f32_16x16x32_f16 v[162:165], a[28:31], a[48:51], v[162:165]
	v_mfma_f32_16x16x32_f16 v[174:177], a[32:35], a[48:51], v[174:177]
	v_mfma_f32_16x16x32_f16 v[98:101], a[0:3], a[48:51], v[98:101]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_f16 v[150:153], a[8:11], a[44:47], v[150:153]
	v_mfma_f32_16x16x32_f16 v[170:173], a[12:15], a[44:47], v[170:173]
	v_mfma_f32_16x16x32_f16 v[166:169], a[16:19], a[44:47], v[166:169]
	v_mfma_f32_16x16x32_f16 v[158:161], a[20:23], a[44:47], v[158:161]
	v_mfma_f32_16x16x32_f16 v[146:149], a[24:27], a[44:47], v[146:149]
	v_mfma_f32_16x16x32_f16 v[130:133], a[28:31], a[44:47], v[130:133]
	v_mfma_f32_16x16x32_f16 v[110:113], a[32:35], a[44:47], v[110:113]
	v_mfma_f32_16x16x32_f16 v[102:105], a[0:3], a[44:47], v[102:105]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[90:93], a[8:11], a[40:43], v[90:93]
	v_mfma_f32_16x16x32_f16 v[114:117], a[12:15], a[40:43], v[114:117]
	v_mfma_f32_16x16x32_f16 v[78:81], a[16:19], a[40:43], v[78:81]
	v_mfma_f32_16x16x32_f16 v[94:97], a[20:23], a[40:43], v[94:97]
	v_mfma_f32_16x16x32_f16 v[74:77], a[24:27], a[40:43], v[74:77]
	v_mfma_f32_16x16x32_f16 v[86:89], a[28:31], a[40:43], v[86:89]
	v_mfma_f32_16x16x32_f16 v[70:73], a[32:35], a[40:43], v[70:73]
	v_mfma_f32_16x16x32_f16 v[82:85], a[0:3], a[40:43], v[82:85]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[106:109], a[8:11], a[36:39], v[106:109]
	v_mfma_f32_16x16x32_f16 v[118:121], a[12:15], a[36:39], v[118:121]
	v_mfma_f32_16x16x32_f16 v[122:125], a[16:19], a[36:39], v[122:125]
	v_mfma_f32_16x16x32_f16 v[126:129], a[20:23], a[36:39], v[126:129]
	v_mfma_f32_16x16x32_f16 a[186:189], a[24:27], a[36:39], a[186:189]
	v_mfma_f32_16x16x32_f16 a[182:185], a[28:31], a[36:39], a[116:119]
	v_mfma_f32_16x16x32_f16 a[178:181], a[32:35], a[36:39], a[112:115]
	v_mfma_f32_16x16x32_f16 v[66:69], a[0:3], a[36:39], v[66:69]
	v_mfma_f32_16x16x32_f16 v[38:41], a[8:11], a[4:7], v[38:41]
	v_mfma_f32_16x16x32_f16 v[62:65], a[12:15], a[4:7], v[62:65]
	v_mfma_f32_16x16x32_f16 v[22:25], a[16:19], a[4:7], v[22:25]
	v_mfma_f32_16x16x32_f16 v[58:61], a[20:23], a[4:7], v[58:61]
	v_mfma_f32_16x16x32_f16 a[158:161], a[24:27], a[4:7], a[64:67]
	v_mfma_f32_16x16x32_f16 a[174:177], a[28:31], a[4:7], a[80:83]
	v_mfma_f32_16x16x32_f16 a[8:11], a[32:35], a[4:7], a[72:75]
	v_mfma_f32_16x16x32_f16 a[166:169], a[0:3], a[4:7], a[68:71]
	s_cbranch_scc1 .LBB0_5
; %bb.6:                                ; %Flow
	v_accvgpr_mov_b32 a142, a170
	v_accvgpr_mov_b32 a138, a154
	s_nop 3
	v_accvgpr_mov_b32 a137, a11
	v_accvgpr_read_b32 v10, a146
	v_accvgpr_read_b32 v249, a153
	v_accvgpr_mov_b32 a143, a171
	v_accvgpr_mov_b32 a144, a172
	v_accvgpr_mov_b32 a145, a173
	v_accvgpr_mov_b32 a139, a155
	v_accvgpr_mov_b32 a140, a156
	v_accvgpr_mov_b32 a141, a157
	v_accvgpr_mov_b32 a136, a10
	v_accvgpr_mov_b32 a135, a9
	v_accvgpr_mov_b32 a134, a8
	v_mov_b32_e32 v1, s16
	v_accvgpr_read_b32 v0, a124
	v_accvgpr_read_b32 v3, a125
	v_accvgpr_read_b32 v14, a126
	v_accvgpr_read_b32 v11, a147
	v_accvgpr_read_b32 v12, a148
	v_accvgpr_read_b32 v13, a149
	v_accvgpr_read_b32 v248, a152
	v_accvgpr_read_b32 v247, a151
	v_accvgpr_read_b32 v246, a150
.LBB0_7:                                ; %Flow265
	s_lshr_b32 s1, s10, 31
	s_add_i32 s1, s10, s1
	s_and_b32 s1, s1, -2
	s_sub_i32 s1, s10, s1
	s_lshl_b32 s1, s1, 14
	s_lshr_b32 s2, s1, 5
	s_add_i32 s2, s2, s1
	s_lshl1_add_u32 s1, s2, 0
	v_and_b32_e32 v254, 15, v0
	v_add_u32_e32 v0, s1, v1
	v_add_u32_e32 v0, 0x107e0, v0
	v_add_u32_e32 v0, v0, v3
	v_lshlrev_b32_e32 v1, 10, v254
	v_lshlrev_b32_e32 v2, 5, v254
	v_add3_u32 v0, v0, v1, v2
	v_add_u32_e32 v1, s1, v1
	v_add3_u32 v1, v1, v14, v2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b128 a[8:11], v1
	ds_read_b128 a[12:15], v1 offset:64
	ds_read_b128 a[16:19], v1 offset:256
	ds_read_b128 a[20:23], v1 offset:320
	ds_read_b128 a[24:27], v1 offset:512
	ds_read_b128 a[28:31], v1 offset:576
	ds_read_b128 a[32:35], v1 offset:768
	ds_read_b128 a[36:39], v1 offset:832
	ds_read_b128 a[40:43], v1 offset:16896
	ds_read_b128 a[44:47], v1 offset:16960
	ds_read_b128 a[48:51], v1 offset:17152
	ds_read_b128 a[52:55], v1 offset:17216
	ds_read_b128 a[56:59], v1 offset:17408
	ds_read_b128 a[60:63], v1 offset:17472
	ds_read_b128 a[64:67], v1 offset:17664
	ds_read_b128 a[0:3], v1 offset:17728
	ds_read_b128 a[68:71], v0
	ds_read_b128 a[72:75], v0 offset:64
	ds_read_b128 a[76:79], v0 offset:256
	ds_read_b128 a[80:83], v0 offset:320
	ds_read_b128 a[84:87], v0 offset:512
	ds_read_b128 a[88:91], v0 offset:576
	ds_read_b128 a[92:95], v0 offset:768
	ds_read_b128 a[96:99], v0 offset:832
	ds_read_b128 a[100:103], v0 offset:16896
	ds_read_b128 a[104:107], v0 offset:16960
	ds_read_b128 a[108:111], v0 offset:17152
	ds_read_b128 a[112:115], v0 offset:17216
	ds_read_b128 a[116:119], v0 offset:17408
	ds_read_b128 a[120:123], v0 offset:17472
	ds_read_b128 a[124:127], v0 offset:17664
	ds_read_b128 a[4:7], v0 offset:17728
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[8:11], v[42:45]
	v_accvgpr_mov_b32 a130, a158
	v_accvgpr_mov_b32 a131, a159
	v_accvgpr_mov_b32 a132, a160
	v_mfma_f32_16x16x32_f16 v[42:45], a[72:75], a[12:15], v[0:3]
	v_accvgpr_mov_b32 a133, a161
	s_lshr_b32 s0, s26, 6
	s_lshl_b32 s0, s0, 3
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[0:3], a[124:127], a[8:11], v[54:57]
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[54:57], a[4:7], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[56:59], v[126:129]
	v_mfma_f32_16x16x32_f16 v[134:137], a[96:99], a[60:63], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[56:59], v[122:125]
	v_mfma_f32_16x16x32_f16 v[126:129], a[88:91], a[60:63], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[56:59], v[118:121]
	v_mfma_f32_16x16x32_f16 v[122:125], a[80:83], a[60:63], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[16:19], v[46:49]
	v_mfma_f32_16x16x32_f16 v[46:49], a[72:75], a[20:23], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[32:35], v[218:221]
	v_mfma_f32_16x16x32_f16 v[142:145], a[72:75], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[24:27], v[242:245]
	v_mfma_f32_16x16x32_f16 v[218:221], a[72:75], a[28:31], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[68:71], a[56:59], v[106:109]
	v_mfma_f32_16x16x32_f16 v[106:109], a[72:75], a[60:63], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[64:67], v[62:65]
	v_mfma_f32_16x16x32_f16 v[242:245], a[80:83], a[0:3], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[48:51], v[114:117]
	v_mfma_f32_16x16x32_f16 v[62:65], a[80:83], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[76:79], a[40:43], v[170:173]
	v_mfma_f32_16x16x32_f16 v[114:117], a[80:83], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[32:35], v[210:213]
	v_mfma_f32_16x16x32_f16 v[170:173], a[88:91], a[36:39], v[0:3]
	s_nop 1
	v_cvt_pk_f16_f32 v212, v42, v43
	v_cvt_pk_f16_f32 v213, v44, v45
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[8:11], v[246:249]
	v_mfma_f32_16x16x32_f16 v[250:253], a[88:91], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[84:87], a[40:43], v[166:169]
	v_mfma_f32_16x16x32_f16 v[138:141], a[88:91], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[64:67], v[58:61]
	v_mfma_f32_16x16x32_f16 v[118:121], a[96:99], a[0:3], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[48:51], v[94:97]
	v_mfma_f32_16x16x32_f16 v[94:97], a[96:99], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[92:95], a[40:43], v[158:161]
	v_mfma_f32_16x16x32_f16 v[158:161], a[96:99], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[32:35], v[186:189]
	v_mfma_f32_16x16x32_f16 v[166:169], a[104:107], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[8:11], v[34:37]
	v_mfma_f32_16x16x32_f16 v[186:189], a[104:107], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[100:103], a[40:43], v[146:149]
	v_mfma_f32_16x16x32_f16 v[14:17], a[124:127], a[48:51], v[82:85]
	v_mfma_f32_16x16x32_f16 v[34:37], a[104:107], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[108:111], a[8:11], v[30:33]
	v_mfma_f32_16x16x32_f16 v[30:33], a[4:7], a[52:55], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[116:119], a[16:19], v[234:237]
	v_mfma_f32_16x16x32_f16 v[146:149], a[112:115], a[12:15], v[0:3]
	s_nop 5
	v_cvt_pk_f16_f32 v43, v32, v33
	v_cvt_pk_f16_f32 v42, v30, v31
	v_mfma_f32_16x16x32_f16 v[0:3], a[108:111], a[48:51], v[86:89]
	v_mfma_f32_16x16x32_f16 v[234:237], a[120:123], a[20:23], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[100:103], a[16:19], v[238:241]
	v_mfma_f32_16x16x32_f16 v[58:61], a[112:115], a[52:55], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[108:111], a[40:43], v[130:133]
	v_mfma_f32_16x16x32_f16 v[238:241], a[104:107], a[20:23], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[84:87], a[16:19], v[6:9]
	v_mfma_f32_16x16x32_f16 v[86:89], a[112:115], a[44:47], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[116:119], a[32:35], v[174:177]
	v_mfma_f32_16x16x32_f16 v[246:249], a[88:91], a[20:23], v[14:17]
	s_nop 5
	v_cvt_pk_f16_f32 v86, v86, v87
	v_cvt_pk_f16_f32 v87, v88, v89
	v_mfma_f32_16x16x32_f16 v[14:17], a[68:71], a[48:51], v[90:93]
	v_mfma_f32_16x16x32_f16 v[130:133], a[120:123], a[36:39], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[116:119], a[8:11], v[26:29]
	v_mfma_f32_16x16x32_f16 v[26:29], a[124:127], a[40:43], v[102:105]
	s_nop 5
	v_cvt_pk_f16_f32 v130, v130, v131
	v_cvt_pk_f16_f32 v131, v132, v133
	v_mfma_f32_16x16x32_f16 v[82:85], a[72:75], a[52:55], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[68:71], a[40:43], v[150:153]
	v_mfma_f32_16x16x32_f16 v[10:13], a[92:95], a[8:11], v[10:13]
	v_mfma_f32_16x16x32_f16 v[50:53], a[4:7], a[44:47], v[26:29]
	v_mfma_f32_16x16x32_f16 v[26:29], a[124:127], a[24:27], v[202:205]
	v_mfma_f32_16x16x32_f16 v[90:93], a[72:75], a[44:47], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[76:79], a[32:35], v[182:185]
	v_mfma_f32_16x16x32_f16 v[182:185], a[96:99], a[12:15], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[100:103], a[48:51], v[74:77]
	v_mfma_f32_16x16x32_f16 v[174:177], a[120:123], a[12:15], v[0:3]
	v_mfma_f32_16x16x32_f16 v[0:3], a[116:119], a[40:43], v[110:113]
	v_mfma_f32_16x16x32_f16 v[102:105], a[4:7], a[28:31], v[26:29]
	v_mfma_f32_16x16x32_f16 v[26:29], a[80:83], a[36:39], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[84:87], a[48:51], v[78:81]
	v_mfma_f32_16x16x32_f16 v[74:77], a[104:107], a[52:55], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[100:103], a[24:27], v[206:209]
	v_mfma_f32_16x16x32_f16 v[110:113], a[120:123], a[44:47], v[0:3]
	s_nop 2
	v_accvgpr_read_b32 v0, a142
	v_accvgpr_read_b32 v1, a143
	v_accvgpr_read_b32 v2, a144
	v_accvgpr_read_b32 v3, a145
	v_mfma_f32_16x16x32_f16 v[78:81], a[88:91], a[52:55], v[14:17]
	v_mfma_f32_16x16x32_f16 v[14:17], a[84:87], a[24:27], v[214:217]
	v_mfma_f32_16x16x32_f16 v[214:217], a[104:107], a[28:31], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[108:111], a[32:35], v[162:165]
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[16:19], v[0:3]
	s_nop 2
	v_accvgpr_read_b32 v0, a138
	v_accvgpr_read_b32 v1, a139
	v_accvgpr_read_b32 v2, a140
	v_accvgpr_read_b32 v3, a141
	v_mfma_f32_16x16x32_f16 v[162:165], a[112:115], a[36:39], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[116:119], a[48:51], v[70:73]
	v_mfma_f32_16x16x32_f16 v[202:205], a[80:83], a[20:23], v[4:7]
	v_mfma_f32_16x16x32_f16 v[4:7], a[76:79], a[8:11], v[0:3]
	v_mfma_f32_16x16x32_f16 v[70:73], a[120:123], a[52:55], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[116:119], a[24:27], v[194:197]
	v_mfma_f32_16x16x32_f16 v[206:209], a[80:83], a[12:15], v[4:7]
	s_nop 5
	v_cvt_pk_f16_f32 v44, v70, v71
	v_cvt_pk_f16_f32 v45, v72, v73
	v_mfma_f32_16x16x32_f16 v[4:7], a[84:87], a[64:67], v[22:25]
	v_mfma_f32_16x16x32_f16 a[158:161], a[100:103], a[56:59], a[186:189]
	v_cvt_pk_f16_f32 v210, v206, v207
	v_cvt_pk_f16_f32 v211, v208, v209
	v_cvt_pk_f16_f32 v208, v250, v251
	v_mfma_f32_16x16x32_f16 v[194:197], a[120:123], a[28:31], v[10:13]
	v_cvt_pk_f16_f32 v209, v252, v253
	v_cvt_pk_f16_f32 v206, v182, v183
	v_cvt_pk_f16_f32 v207, v184, v185
	v_mfma_f32_16x16x32_f16 v[10:13], a[124:127], a[16:19], v[222:225]
	v_cvt_pk_f16_f32 v252, v186, v187
	v_cvt_pk_f16_f32 v253, v188, v189
	v_cvt_pk_f16_f32 v250, v146, v147
	v_mfma_f32_16x16x32_f16 v[22:25], a[88:91], a[0:3], v[4:7]
	v_cvt_pk_f16_f32 v251, v148, v149
	v_cvt_pk_f16_f32 v188, v238, v239
	v_cvt_pk_f16_f32 v189, v240, v241
	v_mfma_f32_16x16x32_f16 v[4:7], a[92:95], a[24:27], v[198:201]
	v_cvt_pk_f16_f32 v184, v234, v235
	v_cvt_pk_f16_f32 v185, v236, v237
	v_cvt_pk_f16_f32 v148, v170, v171
	v_mfma_f32_16x16x32_f16 a[154:157], a[108:111], a[56:59], a[182:185]
	v_cvt_pk_f16_f32 v200, v174, v175
	v_cvt_pk_f16_f32 v201, v176, v177
	v_cvt_pk_f16_f32 v198, v54, v55
	v_mfma_f32_16x16x32_f16 a[150:153], a[116:119], a[56:59], a[178:181]
	v_cvt_pk_f16_f32 v54, v62, v63
	v_cvt_pk_f16_f32 v55, v64, v65
	v_cvt_pk_f16_f32 v199, v56, v57
	v_mfma_f32_16x16x32_f16 a[158:161], a[104:107], a[60:63], a[158:161]
	v_cvt_pk_f16_f32 v149, v172, v173
	v_cvt_pk_f16_f32 v56, v82, v83
	v_cvt_pk_f16_f32 v57, v84, v85
	v_mfma_f32_16x16x32_f16 v[222:225], a[4:7], a[20:23], v[10:13]
	v_mfma_f32_16x16x32_f16 v[10:13], a[108:111], a[16:19], v[230:233]
	v_mfma_f32_16x16x32_f16 v[0:3], a[96:99], a[28:31], v[4:7]
	s_nop 5
	v_cvt_pk_f16_f32 v182, v222, v223
	v_cvt_pk_f16_f32 v183, v224, v225
	v_mfma_f32_16x16x32_f16 a[154:157], a[112:115], a[60:63], a[154:157]
	v_accvgpr_read_b32 v4, a134
	v_accvgpr_read_b32 v5, a135
	v_accvgpr_read_b32 v6, a136
	v_accvgpr_read_b32 v7, a137
	v_mfma_f32_16x16x32_f16 a[150:153], a[120:123], a[60:63], a[150:153]
	v_cvt_pk_f16_f32 v174, v0, v1
	v_cvt_pk_f16_f32 v175, v2, v3
	v_accvgpr_read_b32 v0, a158
	v_mfma_f32_16x16x32_f16 v[230:233], a[112:115], a[20:23], v[10:13]
	v_accvgpr_read_b32 v1, a159
	v_accvgpr_read_b32 v2, a160
	v_accvgpr_read_b32 v3, a161
	v_mfma_f32_16x16x32_f16 v[8:11], a[116:119], a[64:67], v[4:7]
	v_cvt_pk_f16_f32 v32, v0, v1
	v_cvt_pk_f16_f32 v33, v2, v3
	v_accvgpr_read_b32 v0, a154
	v_accvgpr_read_b32 v4, a130
	v_accvgpr_read_b32 v5, a131
	v_accvgpr_read_b32 v6, a132
	v_accvgpr_read_b32 v7, a133
	v_mfma_f32_16x16x32_f16 v[150:153], a[88:91], a[28:31], v[14:17]
	v_accvgpr_read_b32 v1, a155
	v_accvgpr_read_b32 v2, a156
	v_accvgpr_read_b32 v3, a157
	v_mfma_f32_16x16x32_f16 a[8:11], a[108:111], a[64:67], a[174:177]
	v_cvt_pk_f16_f32 v30, v0, v1
	v_cvt_pk_f16_f32 v31, v2, v3
	v_accvgpr_read_b32 v0, a150
	v_mfma_f32_16x16x32_f16 v[12:15], a[68:71], a[64:67], v[38:41]
	v_accvgpr_read_b32 v1, a151
	v_accvgpr_read_b32 v2, a152
	v_accvgpr_read_b32 v3, a153
	v_mfma_f32_16x16x32_f16 v[4:7], a[100:103], a[64:67], v[4:7]
	v_cvt_pk_f16_f32 v176, v150, v151
	v_cvt_pk_f16_f32 v151, v28, v29
	v_cvt_pk_f16_f32 v28, v0, v1
	v_mfma_f32_16x16x32_f16 v[16:19], a[124:127], a[56:59], v[66:69]
	v_cvt_pk_f16_f32 v29, v2, v3
	v_cvt_pk_f16_f32 v150, v26, v27
	v_cvt_pk_f16_f32 v186, v230, v231
	v_mfma_f32_16x16x32_f16 a[8:11], a[112:115], a[0:3], a[8:11]
	v_cvt_pk_f16_f32 v187, v232, v233
	v_cvt_pk_f16_f32 v177, v152, v153
	v_cvt_pk_f16_f32 v152, v142, v143
	v_mfma_f32_16x16x32_f16 v[12:15], a[72:75], a[0:3], v[12:15]
	v_cvt_pk_f16_f32 v153, v144, v145
	v_cvt_pk_f16_f32 v144, v166, v167
	v_cvt_pk_f16_f32 v145, v168, v169
	v_mfma_f32_16x16x32_f16 v[0:3], a[104:107], a[0:3], v[4:7]
	v_cvt_pk_f16_f32 v142, v162, v163
	v_cvt_pk_f16_f32 v143, v164, v165
	v_mfma_f32_16x16x32_f16 a[146:149], a[124:127], a[64:67], a[166:169]
	s_nop 0
	v_cvt_pk_f16_f32 v20, v12, v13
	v_cvt_pk_f16_f32 v21, v14, v15
	v_cvt_pk_f16_f32 v14, v118, v119
	v_mfma_f32_16x16x32_f16 v[16:19], a[4:7], a[60:63], v[16:19]
	v_cvt_pk_f16_f32 v12, v0, v1
	v_cvt_pk_f16_f32 v13, v2, v3
	v_accvgpr_read_b32 v0, a8
	v_accvgpr_read_b32 v1, a9
	v_accvgpr_read_b32 v2, a10
	v_accvgpr_read_b32 v3, a11
	v_mfma_f32_16x16x32_f16 v[8:11], a[120:123], a[0:3], v[8:11]
	v_cvt_pk_f16_f32 v15, v120, v121
	v_cvt_pk_f16_f32 v26, v16, v17
	v_cvt_pk_f16_f32 v16, v22, v23
	v_cvt_pk_f16_f32 v22, v0, v1
	v_cvt_pk_f16_f32 v23, v2, v3
	v_accvgpr_read_b32 v0, a146
	v_accvgpr_read_b32 v1, a147
	v_accvgpr_read_b32 v2, a148
	v_accvgpr_read_b32 v3, a149
	v_cvt_pk_f16_f32 v17, v24, v25
	v_accvgpr_read_b32 v24, a128
	v_mfma_f32_16x16x32_f16 v[0:3], a[4:7], a[0:3], v[0:3]
	v_lshrrev_b32_e32 v24, 2, v24
	v_and_b32_e32 v24, 28, v24
	v_cvt_pk_f16_f32 v4, v8, v9
	v_mfma_f32_16x16x32_f16 v[38:41], a[92:95], a[16:19], v[226:229]
	v_cvt_pk_f16_f32 v5, v10, v11
	v_or_b32_e32 v25, 32, v24
	v_or_b32_e32 v62, 0xc0, v24
	v_cvt_pk_f16_f32 v228, v46, v47
	v_cvt_pk_f16_f32 v0, v0, v1
	v_cvt_pk_f16_f32 v1, v2, v3
	v_and_or_b32 v2, s0, 16, v254
	s_mul_i32 s0, s15, s13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s6, s0
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[0:1], s[14:15], 1
	v_cvt_pk_f16_f32 v46, v58, v59
	v_cvt_pk_f16_f32 v47, v60, v61
	v_or_b32_e32 v3, 32, v2
	v_or_b32_e32 v6, 64, v2
	v_or_b32_e32 v7, 0x60, v2
	v_or_b32_e32 v8, 0x80, v2
	v_or_b32_e32 v9, 0xa0, v2
	v_or_b32_e32 v10, 0xc0, v2
	v_or_b32_e32 v11, 0xe0, v2
	v_or_b32_e32 v58, 64, v24
	v_or_b32_e32 v59, 0x60, v24
	v_or_b32_e32 v60, 0x80, v24
	v_or_b32_e32 v61, 0xa0, v24
	v_or_b32_e32 v63, 0xe0, v24
	s_add_u32 s36, s2, s0
	v_mul_lo_u32 v64, v2, s13
	v_cmp_gt_i32_e64 s[28:29], s8, v2
	v_cmp_gt_i32_e64 s[14:15], s9, v24
	v_cvt_pk_f16_f32 v229, v48, v49
	v_cvt_pk_f16_f32 v48, v74, v75
	s_addc_u32 s33, s3, s1
	v_mul_lo_u32 v65, v3, s13
	v_mul_lo_u32 v70, v6, s13
	v_mul_lo_u32 v71, v7, s13
	v_mul_lo_u32 v72, v8, s13
	v_mul_lo_u32 v73, v9, s13
	v_mul_lo_u32 v74, v10, s13
	v_mul_lo_u32 v75, v11, s13
	v_cmp_gt_i32_e64 s[26:27], s8, v3
	v_cmp_gt_i32_e64 s[24:25], s8, v6
	v_cmp_gt_i32_e64 s[22:23], s8, v7
	v_cmp_gt_i32_e64 s[20:21], s8, v8
	v_cmp_gt_i32_e64 s[18:19], s8, v9
	v_cmp_gt_i32_e64 s[16:17], s8, v10
	v_cmp_gt_i32_e32 vcc, s8, v11
	v_cmp_gt_i32_e64 s[12:13], s9, v25
	v_cmp_gt_i32_e64 s[10:11], s9, v58
	v_cmp_gt_i32_e64 s[30:31], s9, v59
	v_cmp_gt_i32_e64 s[6:7], s9, v60
	v_cmp_gt_i32_e64 s[4:5], s9, v61
	v_cmp_gt_i32_e64 s[2:3], s9, v62
	v_cmp_gt_i32_e64 s[0:1], s9, v63
	v_add_lshl_u32 v2, v24, v64, 1
	v_bfrev_b32_e32 v3, 1
	s_and_b64 s[8:9], s[28:29], s[14:15]
	s_and_b32 s37, s33, 0xffff
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[212:213], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v25, v64, 1
	s_and_b64 s[8:9], s[28:29], s[12:13]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[210:211], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v58, v64, 1
	s_and_b64 s[8:9], s[28:29], s[10:11]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[208:209], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v59, v64, 1
	s_and_b64 s[8:9], s[28:29], s[30:31]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[206:207], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v60, v64, 1
	s_and_b64 s[8:9], s[28:29], s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[252:253], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v61, v64, 1
	s_and_b64 s[8:9], s[28:29], s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[250:251], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v62, v64, 1
	s_and_b64 s[8:9], s[28:29], s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[200:201], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v63, v64, 1
	s_and_b64 s[8:9], s[28:29], s[0:1]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[198:199], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v65, v24, 1
	s_and_b64 s[8:9], s[26:27], s[14:15]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[228:229], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v25, v65, 1
	s_and_b64 s[8:9], s[26:27], s[12:13]
	v_mfma_f32_16x16x32_f16 v[38:41], a[96:99], a[20:23], v[38:41]
	v_cvt_pk_f16_f32 v226, v202, v203
	v_cvt_pk_f16_f32 v227, v204, v205
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[226:227], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v58, v65, 1
	s_and_b64 s[8:9], s[26:27], s[10:11]
	v_mfma_f32_16x16x32_f16 v[66:69], a[92:95], a[32:35], v[154:157]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	s_and_b64 s[8:9], s[26:27], s[30:31]
	v_cvt_pk_f16_f32 v202, v218, v219
	v_mfma_f32_16x16x32_f16 v[154:157], a[108:111], a[24:27], v[178:181]
	v_cvt_pk_f16_f32 v203, v220, v221
	v_cvt_pk_f16_f32 v218, v214, v215
	v_cvt_pk_f16_f32 v219, v216, v217
	v_mfma_f32_16x16x32_f16 v[178:181], a[76:79], a[24:27], v[190:193]
	v_cvt_pk_f16_f32 v49, v76, v77
	v_cvt_pk_f16_f32 v27, v18, v19
	v_cvt_pk_f16_f32 v18, v242, v243
	v_cvt_pk_f16_f32 v192, v246, v247
	v_cvt_pk_f16_f32 v193, v248, v249
	buffer_store_dwordx2 v[192:193], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v59, v65, 1
	v_cvt_pk_f16_f32 v190, v38, v39
	v_cvt_pk_f16_f32 v191, v40, v41
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[190:191], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v60, v65, 1
	s_and_b64 s[8:9], s[26:27], s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[188:189], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v61, v65, 1
	s_and_b64 s[8:9], s[26:27], s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[186:187], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v62, v65, 1
	s_and_b64 s[8:9], s[26:27], s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[184:185], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v63, v65, 1
	s_and_b64 s[8:9], s[26:27], s[0:1]
	v_mfma_f32_16x16x32_f16 v[178:181], a[80:83], a[28:31], v[178:181]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[182:183], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v70, v24, 1
	s_and_b64 s[8:9], s[24:25], s[14:15]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[202:203], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v25, v70, 1
	s_and_b64 s[8:9], s[24:25], s[12:13]
	v_cvt_pk_f16_f32 v178, v178, v179
	v_cvt_pk_f16_f32 v179, v180, v181
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[178:179], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v58, v70, 1
	s_and_b64 s[8:9], s[24:25], s[10:11]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[176:177], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v59, v70, 1
	s_and_b64 s[8:9], s[24:25], s[30:31]
	v_mfma_f32_16x16x32_f16 v[154:157], a[112:115], a[28:31], v[154:157]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[174:175], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v60, v70, 1
	s_and_b64 s[8:9], s[24:25], s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[218:219], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v61, v70, 1
	s_and_b64 s[8:9], s[24:25], s[4:5]
	v_cvt_pk_f16_f32 v214, v154, v155
	v_cvt_pk_f16_f32 v215, v156, v157
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[214:215], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v62, v70, 1
	s_and_b64 s[8:9], s[24:25], s[2:3]
	v_cvt_pk_f16_f32 v156, v194, v195
	v_cvt_pk_f16_f32 v157, v196, v197
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[156:157], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v63, v70, 1
	s_and_b64 s[8:9], s[24:25], s[0:1]
	v_cvt_pk_f16_f32 v154, v102, v103
	v_cvt_pk_f16_f32 v155, v104, v105
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[154:155], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v71, v24, 1
	s_and_b64 s[8:9], s[22:23], s[14:15]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[152:153], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v25, v71, 1
	s_and_b64 s[8:9], s[22:23], s[12:13]
	v_mfma_f32_16x16x32_f16 v[66:69], a[96:99], a[36:39], v[66:69]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[150:151], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v58, v71, 1
	s_and_b64 s[8:9], s[22:23], s[10:11]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[148:149], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v59, v71, 1
	s_and_b64 s[8:9], s[22:23], s[30:31]
	v_mfma_f32_16x16x32_f16 v[98:101], a[124:127], a[32:35], v[98:101]
	v_cvt_pk_f16_f32 v146, v66, v67
	v_cvt_pk_f16_f32 v147, v68, v69
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[146:147], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v60, v71, 1
	s_and_b64 s[8:9], s[22:23], s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[144:145], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v61, v71, 1
	s_and_b64 s[8:9], s[22:23], s[4:5]
	v_mfma_f32_16x16x32_f16 v[98:101], a[4:7], a[36:39], v[98:101]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[142:143], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v62, v71, 1
	s_and_b64 s[8:9], s[22:23], s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[130:131], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v63, v71, 1
	s_and_b64 s[8:9], s[22:23], s[0:1]
	v_cvt_pk_f16_f32 v104, v98, v99
	v_cvt_pk_f16_f32 v105, v100, v101
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[104:105], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v72, v24, 1
	s_and_b64 s[8:9], s[20:21], s[14:15]
	v_cvt_pk_f16_f32 v102, v90, v91
	v_cvt_pk_f16_f32 v103, v92, v93
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[102:103], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v25, v72, 1
	s_and_b64 s[8:9], s[20:21], s[12:13]
	v_cvt_pk_f16_f32 v100, v114, v115
	v_cvt_pk_f16_f32 v101, v116, v117
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[100:101], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v58, v72, 1
	s_and_b64 s[8:9], s[20:21], s[10:11]
	v_cvt_pk_f16_f32 v98, v138, v139
	v_cvt_pk_f16_f32 v99, v140, v141
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[98:99], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v59, v72, 1
	s_and_b64 s[8:9], s[20:21], s[30:31]
	v_cvt_pk_f16_f32 v92, v158, v159
	v_cvt_pk_f16_f32 v93, v160, v161
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[92:93], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v60, v72, 1
	s_and_b64 s[8:9], s[20:21], s[6:7]
	v_cvt_pk_f16_f32 v90, v34, v35
	v_cvt_pk_f16_f32 v91, v36, v37
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[90:91], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v61, v72, 1
	s_and_b64 s[8:9], s[20:21], s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[86:87], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v62, v72, 1
	s_and_b64 s[8:9], s[20:21], s[2:3]
	v_cvt_pk_f16_f32 v68, v110, v111
	v_cvt_pk_f16_f32 v69, v112, v113
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[68:69], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v63, v72, 1
	s_and_b64 s[8:9], s[20:21], s[0:1]
	v_cvt_pk_f16_f32 v66, v50, v51
	v_cvt_pk_f16_f32 v67, v52, v53
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[66:67], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v73, v24, 1
	s_and_b64 s[8:9], s[18:19], s[14:15]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[56:57], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v25, v73, 1
	s_and_b64 s[8:9], s[18:19], s[12:13]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[54:55], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v58, v73, 1
	s_and_b64 s[8:9], s[18:19], s[10:11]
	v_cvt_pk_f16_f32 v52, v78, v79
	v_cvt_pk_f16_f32 v53, v80, v81
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[52:53], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v59, v73, 1
	s_and_b64 s[8:9], s[18:19], s[30:31]
	v_cvt_pk_f16_f32 v50, v94, v95
	v_cvt_pk_f16_f32 v51, v96, v97
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[50:51], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v60, v73, 1
	s_and_b64 s[8:9], s[18:19], s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[48:49], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v61, v73, 1
	s_and_b64 s[8:9], s[18:19], s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[46:47], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v62, v73, 1
	s_and_b64 s[8:9], s[18:19], s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[44:45], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v63, v73, 1
	s_and_b64 s[8:9], s[18:19], s[0:1]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[42:43], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v74, v24, 1
	s_and_b64 s[8:9], s[16:17], s[14:15]
	v_cvt_pk_f16_f32 v40, v106, v107
	v_cvt_pk_f16_f32 v41, v108, v109
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[40:41], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v25, v74, 1
	s_and_b64 s[8:9], s[16:17], s[12:13]
	v_cvt_pk_f16_f32 v38, v122, v123
	v_cvt_pk_f16_f32 v39, v124, v125
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[38:39], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v58, v74, 1
	s_and_b64 s[8:9], s[16:17], s[10:11]
	v_cvt_pk_f16_f32 v36, v126, v127
	v_cvt_pk_f16_f32 v37, v128, v129
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[36:37], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v59, v74, 1
	s_and_b64 s[8:9], s[16:17], s[30:31]
	v_cvt_pk_f16_f32 v34, v134, v135
	v_cvt_pk_f16_f32 v35, v136, v137
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[34:35], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v60, v74, 1
	s_and_b64 s[8:9], s[16:17], s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[32:33], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v61, v74, 1
	s_and_b64 s[8:9], s[16:17], s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[30:31], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v62, v74, 1
	s_and_b64 s[8:9], s[16:17], s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[28:29], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v63, v74, 1
	s_and_b64 s[8:9], s[16:17], s[0:1]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[26:27], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v75, v24, 1
	s_and_b64 s[8:9], vcc, s[14:15]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[20:21], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v25, v75, 1
	s_and_b64 s[8:9], vcc, s[12:13]
	v_cvt_pk_f16_f32 v19, v244, v245
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[18:19], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v58, v75, 1
	s_and_b64 s[8:9], vcc, s[10:11]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[16:17], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v59, v75, 1
	s_and_b64 s[8:9], vcc, s[30:31]
	v_cndmask_b32_e64 v2, v3, v2, s[8:9]
	buffer_store_dwordx2 v[14:15], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v60, v75, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_cndmask_b32_e64 v2, v3, v2, s[6:7]
	buffer_store_dwordx2 v[12:13], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v61, v75, 1
	s_and_b64 s[4:5], vcc, s[4:5]
	v_cndmask_b32_e64 v2, v3, v2, s[4:5]
	buffer_store_dwordx2 v[22:23], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v62, v75, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_cndmask_b32_e64 v2, v3, v2, s[2:3]
	buffer_store_dwordx2 v[4:5], v2, s[36:39], 0 offen
	v_add_lshl_u32 v2, v63, v75, 1
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v2, v3, v2, vcc
	buffer_store_dwordx2 v[0:1], v2, s[36:39], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v4_global_prefetch
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
		.amdhsa_next_free_vgpr 446
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
	.size	v4_global_prefetch, .Lfunc_end0-v4_global_prefetch
	.cfi_endproc
                                        ; -- End function
	.set v4_global_prefetch.num_vgpr, 256
	.set v4_global_prefetch.num_agpr, 190
	.set v4_global_prefetch.numbered_sgpr, 40
	.set v4_global_prefetch.num_named_barrier, 0
	.set v4_global_prefetch.private_seg_size, 0
	.set v4_global_prefetch.uses_vcc, 1
	.set v4_global_prefetch.uses_flat_scratch, 0
	.set v4_global_prefetch.has_dyn_sized_stack, 0
	.set v4_global_prefetch.has_recursion, 0
	.set v4_global_prefetch.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11616
; TotalNumSgprs: 46
; NumVgprs: 256
; NumAgprs: 190
; TotalNumVgprs: 446
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 55
; NumSGPRsForWavesPerEU: 46
; NumVGPRsForWavesPerEU: 446
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
	.byte	48                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	136                             ; DW_AT_call_line
	.byte	25                              ; DW_AT_call_column
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
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v4_global_prefetch" ; string offset=24
.Linfo_string3:
	.asciz	"v4_global_prefetch"            ; string offset=90
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     190
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
    .name:           v4_global_prefetch
    .private_segment_fixed_size: 0
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         v4_global_prefetch.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     446
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
