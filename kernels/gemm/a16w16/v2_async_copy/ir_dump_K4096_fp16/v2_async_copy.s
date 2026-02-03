	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v2_async_copy                   ; -- Begin function v2_async_copy
	.p2align	8
	.type	v2_async_copy,@function
v2_async_copy:                          ; @v2_async_copy
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.7:
	.file	1 "/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v2_async_copy" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.8:
.LBB0_0:
	.file	2 "/root/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s9, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s14, s0, 8
	s_abs_i32 s19, s16
	s_xor_b32 s18, s16, s14
	s_ashr_i32 s18, s18, 31
	v_readfirstlane_b32 s0, v0
	s_bfe_u32 s1, s0, 0x20006
	s_abs_i32 s17, s14
	s_sub_i32 s20, 0, s17
	s_lshr_b32 s15, s0, 6
	v_cvt_f32_u32_e32 v1, s17
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a131, 0
	v_rcp_iflag_f32_e32 v1, v1
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a136, 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a134, 0
	v_mov_b32_e32 v13, 0
	v_readfirstlane_b32 s21, v1
	s_mul_i32 s20, s20, s21
	s_mul_hi_u32 s20, s21, s20
	s_add_i32 s21, s21, s20
	s_mul_hi_u32 s20, s19, s21
	s_mul_i32 s21, s20, s17
	s_sub_i32 s19, s19, s21
	s_add_i32 s21, s20, 1
	s_sub_i32 s22, s19, s17
	s_cmp_ge_u32 s19, s17
	s_cselect_b32 s20, s21, s20
	s_cselect_b32 s19, s22, s19
	s_add_i32 s21, s20, 1
	s_cmp_ge_u32 s19, s17
	s_cselect_b32 s17, s21, s20
	s_xor_b32 s17, s17, s18
	s_sub_i32 s17, s17, s18
	s_mul_i32 s14, s17, s14
	s_sub_i32 s14, s16, s14
	v_and_b32_e32 v1, 63, v0
	v_lshl_or_b32 v1, s1, 6, v1
	s_lshl_b32 s16, s17, 8
	s_lshl_b32 s14, s14, 8
	s_add_i32 s17, s10, 63
	v_accvgpr_write_b32 a129, v1
	s_cmp_lt_i32 s17, 64
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	s_cbranch_scc1 .LBB0_6
; %bb.1:                                ; %.lr.ph
	v_lshlrev_b32_e32 v1, 3, v0
	v_and_b32_e32 v254, 56, v1
	v_accvgpr_read_b32 v1, a129
	v_lshrrev_b32_e32 v16, 3, v1
	v_mov_b32_e32 v17, v0
	v_or_b32_e32 v13, 32, v16
	v_mul_lo_u32 v0, v16, s11
	v_or_b32_e32 v11, 64, v16
	v_mul_lo_u32 v14, v13, s12
	v_mul_lo_u32 v13, v13, s11
	v_add_lshl_u32 v0, v0, v254, 1
	v_or_b32_e32 v9, 0x60, v16
	v_mul_lo_u32 v12, v11, s12
	v_mul_lo_u32 v11, v11, s11
	v_accvgpr_write_b32 a138, v0
	v_add_lshl_u32 v0, v13, v254, 1
	v_or_b32_e32 v7, 0x80, v16
	v_mul_lo_u32 v10, v9, s12
	v_mul_lo_u32 v9, v9, s11
	v_accvgpr_write_b32 a139, v0
	v_add_lshl_u32 v0, v11, v254, 1
	s_ashr_i32 s18, s17, 31
	v_or_b32_e32 v5, 0xa0, v16
	v_mul_lo_u32 v8, v7, s12
	v_mul_lo_u32 v7, v7, s11
	v_accvgpr_write_b32 a140, v0
	v_add_lshl_u32 v0, v9, v254, 1
	s_lshr_b32 s18, s18, 26
	v_or_b32_e32 v3, 0xc0, v16
	v_mul_lo_u32 v6, v5, s12
	v_mul_lo_u32 v5, v5, s11
	v_accvgpr_write_b32 a141, v0
	v_add_lshl_u32 v0, v7, v254, 1
	s_add_i32 s17, s17, s18
	v_or_b32_e32 v1, 0xe0, v16
	v_mul_lo_u32 v4, v3, s12
	v_mul_lo_u32 v3, v3, s11
	s_mul_i32 s18, s14, s12
	v_accvgpr_write_b32 a142, v0
	v_add_lshl_u32 v0, v5, v254, 1
	v_mul_lo_u32 v2, v1, s12
	v_mul_lo_u32 v1, v1, s11
	s_ashr_i32 s19, s18, 31
	v_accvgpr_write_b32 a143, v0
	v_add_lshl_u32 v0, v3, v254, 1
	s_ashr_i32 s17, s17, 6
	v_mul_lo_u32 v15, v16, s12
	s_lshl_b64 s[18:19], s[18:19], 1
	v_accvgpr_write_b32 a144, v0
	v_add_lshl_u32 v0, v1, v254, 1
	s_add_u32 s12, s4, s18
	s_mul_i32 s4, s16, s11
	v_accvgpr_write_b32 a145, v0
	v_add_lshl_u32 v0, v15, v254, 1
	s_addc_u32 s18, s5, s19
	s_ashr_i32 s5, s4, 31
	v_accvgpr_write_b32 a146, v0
	v_add_lshl_u32 v0, v14, v254, 1
	s_lshl_b64 s[4:5], s[4:5], 1
	v_accvgpr_write_b32 a147, v0
	v_add_lshl_u32 v0, v12, v254, 1
	s_add_u32 s11, s2, s4
	v_accvgpr_write_b32 a148, v0
	v_add_lshl_u32 v0, v10, v254, 1
	s_addc_u32 s19, s3, s5
	v_accvgpr_write_b32 a149, v0
	v_add_lshl_u32 v0, v8, v254, 1
	v_accvgpr_write_b32 a150, v0
	v_add_lshl_u32 v0, v6, v254, 1
	v_accvgpr_write_b32 a151, v0
	v_add_lshl_u32 v0, v2, v254, 1
	v_lshlrev_b32_e32 v1, 7, v17
	v_and_b32_e32 v2, 48, v17
	s_lshl_b32 s1, s1, 10
	s_add_i32 s20, s1, 0
	v_add_lshl_u32 v255, v4, v254, 1
	s_add_i32 s36, s20, 0xf000
	s_lshl_b32 s0, s0, 5
	s_and_b32 s0, s0, 0x800
	v_accvgpr_write_b32 a128, v17
	s_add_i32 s38, s0, 0
	s_lshl_b32 s1, s15, 10
	s_and_b32 s1, s1, 0x800
	s_add_i32 s37, s1, 0
	s_add_i32 s21, s20, 0x1000
	s_add_i32 s22, s20, 0x2000
	s_add_i32 s23, s20, 0x3000
	s_add_i32 s24, s20, 0x4000
	s_add_i32 s25, s20, 0x5000
	s_add_i32 s26, s20, 0x6000
	s_add_i32 s27, s20, 0x7000
	s_add_i32 s28, s20, 0x8000
	s_add_i32 s29, s20, 0x9000
	s_add_i32 s30, s20, 0xa000
	s_add_i32 s31, s20, 0xb000
	s_add_i32 s33, s20, 0xc000
	s_add_i32 s34, s20, 0xd000
	s_add_i32 s35, s20, 0xe000
	v_mov_b32_e32 v3, 0
	s_movk_i32 s2, 0x780
	v_and_or_b32 v1, v1, s2, v2
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a133, 0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	s_branch .LBB0_3
.LBB0_2:                                ; %.critedge26
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[4:5]
	v_add_u32_e32 v6, s37, v1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b128 a[80:83], v6
	ds_read_b128 a[84:87], v6 offset:64
	ds_read_b128 a[88:91], v6 offset:4096
	ds_read_b128 a[92:95], v6 offset:4160
	ds_read_b128 a[96:99], v6 offset:8192
	ds_read_b128 a[100:103], v6 offset:8256
	ds_read_b128 a[104:107], v6 offset:12288
	ds_read_b128 a[108:111], v6 offset:12352
	ds_read_b128 a[112:115], v6 offset:16384
	ds_read_b128 a[116:119], v6 offset:16448
	ds_read_b128 a[120:123], v6 offset:20480
	ds_read_b128 a[124:127], v6 offset:20544
	ds_read_b128 a[76:79], v6 offset:24576
	ds_read_b128 a[72:75], v6 offset:24640
	ds_read_b128 a[0:3], v6 offset:28672
	ds_read_b128 a[4:7], v6 offset:28736
	v_add_u32_e32 v6, s38, v1
	ds_read_b128 a[64:67], v6 offset:32768
	ds_read_b128 a[68:71], v6 offset:32832
	ds_read_b128 a[56:59], v6 offset:36864
	ds_read_b128 a[60:63], v6 offset:36928
	ds_read_b128 a[48:51], v6 offset:40960
	ds_read_b128 a[52:55], v6 offset:41024
	ds_read_b128 a[40:43], v6 offset:45056
	ds_read_b128 a[44:47], v6 offset:45120
	ds_read_b128 a[32:35], v6 offset:49152
	ds_read_b128 a[36:39], v6 offset:49216
	ds_read_b128 a[24:27], v6 offset:53248
	ds_read_b128 a[28:31], v6 offset:53312
	ds_read_b128 a[8:11], v6 offset:57344
	ds_read_b128 a[12:15], v6 offset:57408
	ds_read_b128 a[16:19], v6 offset:61440
	ds_read_b128 a[20:23], v6 offset:61504
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_16x16x32_f16 v[2:5], a[64:67], a[80:83], v[2:5]
	s_add_u32 s11, s11, 0x80
	s_addc_u32 s19, s19, 0
	s_add_u32 s12, s12, 0x80
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_16x16x32_f16 v[250:253], a[56:59], a[80:83], v[250:253]
	s_addc_u32 s18, s18, 0
	s_add_i32 s17, s17, -1
	s_sub_i32 s10, s10, 64
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x32_f16 v[246:249], a[48:51], a[80:83], v[246:249]
	s_cmp_lg_u32 s17, 0
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_f16 v[242:245], a[40:43], a[80:83], v[242:245]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 v[238:241], a[32:35], a[80:83], v[238:241]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_f16 v[234:237], a[24:27], a[80:83], v[234:237]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 v[230:233], a[8:11], a[80:83], v[230:233]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[226:229], a[16:19], a[80:83], v[226:229]
	v_mfma_f32_16x16x32_f16 v[222:225], a[64:67], a[88:91], v[222:225]
	v_mfma_f32_16x16x32_f16 v[218:221], a[56:59], a[88:91], v[218:221]
	v_mfma_f32_16x16x32_f16 v[214:217], a[48:51], a[88:91], v[214:217]
	v_mfma_f32_16x16x32_f16 v[210:213], a[40:43], a[88:91], v[210:213]
	v_mfma_f32_16x16x32_f16 v[206:209], a[32:35], a[88:91], v[206:209]
	v_mfma_f32_16x16x32_f16 v[198:201], a[24:27], a[88:91], v[198:201]
	v_mfma_f32_16x16x32_f16 v[202:205], a[8:11], a[88:91], v[202:205]
	v_mfma_f32_16x16x32_f16 v[194:197], a[16:19], a[88:91], v[194:197]
	v_mfma_f32_16x16x32_f16 v[190:193], a[64:67], a[96:99], v[190:193]
	v_mfma_f32_16x16x32_f16 v[186:189], a[56:59], a[96:99], v[186:189]
	v_mfma_f32_16x16x32_f16 v[182:185], a[48:51], a[96:99], v[182:185]
	v_mfma_f32_16x16x32_f16 v[178:181], a[40:43], a[96:99], v[178:181]
	v_mfma_f32_16x16x32_f16 v[174:177], a[32:35], a[96:99], v[174:177]
	v_mfma_f32_16x16x32_f16 v[170:173], a[24:27], a[96:99], v[170:173]
	v_mfma_f32_16x16x32_f16 v[166:169], a[8:11], a[96:99], v[166:169]
	v_mfma_f32_16x16x32_f16 v[162:165], a[16:19], a[96:99], v[162:165]
	v_mfma_f32_16x16x32_f16 v[158:161], a[64:67], a[104:107], v[158:161]
	v_mfma_f32_16x16x32_f16 v[154:157], a[56:59], a[104:107], v[154:157]
	v_mfma_f32_16x16x32_f16 v[150:153], a[48:51], a[104:107], v[150:153]
	v_mfma_f32_16x16x32_f16 v[146:149], a[40:43], a[104:107], v[146:149]
	v_mfma_f32_16x16x32_f16 v[142:145], a[32:35], a[104:107], v[142:145]
	v_mfma_f32_16x16x32_f16 v[138:141], a[24:27], a[104:107], v[138:141]
	v_mfma_f32_16x16x32_f16 v[134:137], a[8:11], a[104:107], v[134:137]
	v_mfma_f32_16x16x32_f16 v[130:133], a[16:19], a[104:107], v[130:133]
	v_mfma_f32_16x16x32_f16 v[126:129], a[64:67], a[112:115], v[126:129]
	v_mfma_f32_16x16x32_f16 v[122:125], a[56:59], a[112:115], v[122:125]
	v_mfma_f32_16x16x32_f16 v[118:121], a[48:51], a[112:115], v[118:121]
	v_mfma_f32_16x16x32_f16 v[114:117], a[40:43], a[112:115], v[114:117]
	v_mfma_f32_16x16x32_f16 v[110:113], a[32:35], a[112:115], v[110:113]
	v_mfma_f32_16x16x32_f16 v[106:109], a[24:27], a[112:115], v[106:109]
	v_mfma_f32_16x16x32_f16 v[102:105], a[8:11], a[112:115], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[16:19], a[112:115], v[98:101]
	v_mfma_f32_16x16x32_f16 v[94:97], a[64:67], a[120:123], v[94:97]
	v_mfma_f32_16x16x32_f16 v[90:93], a[56:59], a[120:123], v[90:93]
	v_mfma_f32_16x16x32_f16 v[86:89], a[48:51], a[120:123], v[86:89]
	v_mfma_f32_16x16x32_f16 v[82:85], a[40:43], a[120:123], v[82:85]
	v_mfma_f32_16x16x32_f16 v[78:81], a[32:35], a[120:123], v[78:81]
	v_mfma_f32_16x16x32_f16 v[74:77], a[24:27], a[120:123], v[74:77]
	v_mfma_f32_16x16x32_f16 v[70:73], a[8:11], a[120:123], v[70:73]
	v_mfma_f32_16x16x32_f16 v[66:69], a[16:19], a[120:123], v[66:69]
	v_mfma_f32_16x16x32_f16 v[62:65], a[64:67], a[76:79], v[62:65]
	v_mfma_f32_16x16x32_f16 v[58:61], a[56:59], a[76:79], v[58:61]
	v_mfma_f32_16x16x32_f16 v[54:57], a[48:51], a[76:79], v[54:57]
	v_mfma_f32_16x16x32_f16 v[50:53], a[40:43], a[76:79], v[50:53]
	v_mfma_f32_16x16x32_f16 v[46:49], a[32:35], a[76:79], v[46:49]
	v_mfma_f32_16x16x32_f16 v[42:45], a[24:27], a[76:79], v[42:45]
	v_mfma_f32_16x16x32_f16 v[38:41], a[8:11], a[76:79], v[38:41]
	v_mfma_f32_16x16x32_f16 v[34:37], a[16:19], a[76:79], v[34:37]
	v_mfma_f32_16x16x32_f16 v[30:33], a[64:67], a[0:3], v[30:33]
	v_mfma_f32_16x16x32_f16 v[26:29], a[56:59], a[0:3], v[26:29]
	v_mfma_f32_16x16x32_f16 v[22:25], a[48:51], a[0:3], v[22:25]
	v_mfma_f32_16x16x32_f16 v[18:21], a[40:43], a[0:3], v[18:21]
	v_mfma_f32_16x16x32_f16 v[14:17], a[32:35], a[0:3], v[14:17]
	v_mfma_f32_16x16x32_f16 v[10:13], a[24:27], a[0:3], v[10:13]
	v_mfma_f32_16x16x32_f16 a[134:137], a[8:11], a[0:3], a[134:137]
	v_mfma_f32_16x16x32_f16 a[130:133], a[16:19], a[0:3], a[130:133]
	v_mfma_f32_16x16x32_f16 v[2:5], a[68:71], a[84:87], v[2:5]
	v_mfma_f32_16x16x32_f16 v[250:253], a[60:63], a[84:87], v[250:253]
	v_mfma_f32_16x16x32_f16 v[246:249], a[52:55], a[84:87], v[246:249]
	v_mfma_f32_16x16x32_f16 v[242:245], a[44:47], a[84:87], v[242:245]
	v_mfma_f32_16x16x32_f16 v[238:241], a[36:39], a[84:87], v[238:241]
	v_mfma_f32_16x16x32_f16 v[234:237], a[28:31], a[84:87], v[234:237]
	v_mfma_f32_16x16x32_f16 v[230:233], a[12:15], a[84:87], v[230:233]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[226:229], a[20:23], a[84:87], v[226:229]
	v_mfma_f32_16x16x32_f16 v[222:225], a[68:71], a[92:95], v[222:225]
	v_mfma_f32_16x16x32_f16 v[218:221], a[60:63], a[92:95], v[218:221]
	v_mfma_f32_16x16x32_f16 v[214:217], a[52:55], a[92:95], v[214:217]
	v_mfma_f32_16x16x32_f16 v[210:213], a[44:47], a[92:95], v[210:213]
	v_mfma_f32_16x16x32_f16 v[206:209], a[36:39], a[92:95], v[206:209]
	v_mfma_f32_16x16x32_f16 v[198:201], a[28:31], a[92:95], v[198:201]
	v_mfma_f32_16x16x32_f16 v[202:205], a[12:15], a[92:95], v[202:205]
	v_mfma_f32_16x16x32_f16 v[194:197], a[20:23], a[92:95], v[194:197]
	v_mfma_f32_16x16x32_f16 v[190:193], a[68:71], a[100:103], v[190:193]
	v_mfma_f32_16x16x32_f16 v[186:189], a[60:63], a[100:103], v[186:189]
	v_mfma_f32_16x16x32_f16 v[182:185], a[52:55], a[100:103], v[182:185]
	v_mfma_f32_16x16x32_f16 v[178:181], a[44:47], a[100:103], v[178:181]
	v_mfma_f32_16x16x32_f16 v[174:177], a[36:39], a[100:103], v[174:177]
	v_mfma_f32_16x16x32_f16 v[170:173], a[28:31], a[100:103], v[170:173]
	v_mfma_f32_16x16x32_f16 v[166:169], a[12:15], a[100:103], v[166:169]
	v_mfma_f32_16x16x32_f16 v[162:165], a[20:23], a[100:103], v[162:165]
	v_mfma_f32_16x16x32_f16 v[158:161], a[68:71], a[108:111], v[158:161]
	v_mfma_f32_16x16x32_f16 v[154:157], a[60:63], a[108:111], v[154:157]
	v_mfma_f32_16x16x32_f16 v[150:153], a[52:55], a[108:111], v[150:153]
	v_mfma_f32_16x16x32_f16 v[146:149], a[44:47], a[108:111], v[146:149]
	v_mfma_f32_16x16x32_f16 v[142:145], a[36:39], a[108:111], v[142:145]
	v_mfma_f32_16x16x32_f16 v[138:141], a[28:31], a[108:111], v[138:141]
	v_mfma_f32_16x16x32_f16 v[134:137], a[12:15], a[108:111], v[134:137]
	v_mfma_f32_16x16x32_f16 v[130:133], a[20:23], a[108:111], v[130:133]
	v_mfma_f32_16x16x32_f16 v[126:129], a[68:71], a[116:119], v[126:129]
	v_mfma_f32_16x16x32_f16 v[122:125], a[60:63], a[116:119], v[122:125]
	v_mfma_f32_16x16x32_f16 v[118:121], a[52:55], a[116:119], v[118:121]
	v_mfma_f32_16x16x32_f16 v[114:117], a[44:47], a[116:119], v[114:117]
	v_mfma_f32_16x16x32_f16 v[110:113], a[36:39], a[116:119], v[110:113]
	v_mfma_f32_16x16x32_f16 v[106:109], a[28:31], a[116:119], v[106:109]
	v_mfma_f32_16x16x32_f16 v[102:105], a[12:15], a[116:119], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[20:23], a[116:119], v[98:101]
	v_mfma_f32_16x16x32_f16 v[94:97], a[68:71], a[124:127], v[94:97]
	v_mfma_f32_16x16x32_f16 v[90:93], a[60:63], a[124:127], v[90:93]
	v_mfma_f32_16x16x32_f16 v[86:89], a[52:55], a[124:127], v[86:89]
	v_mfma_f32_16x16x32_f16 v[82:85], a[44:47], a[124:127], v[82:85]
	v_mfma_f32_16x16x32_f16 v[78:81], a[36:39], a[124:127], v[78:81]
	v_mfma_f32_16x16x32_f16 v[74:77], a[28:31], a[124:127], v[74:77]
	v_mfma_f32_16x16x32_f16 v[70:73], a[12:15], a[124:127], v[70:73]
	v_mfma_f32_16x16x32_f16 v[66:69], a[20:23], a[124:127], v[66:69]
	v_mfma_f32_16x16x32_f16 v[62:65], a[68:71], a[72:75], v[62:65]
	v_mfma_f32_16x16x32_f16 v[58:61], a[60:63], a[72:75], v[58:61]
	v_mfma_f32_16x16x32_f16 v[54:57], a[52:55], a[72:75], v[54:57]
	v_mfma_f32_16x16x32_f16 v[50:53], a[44:47], a[72:75], v[50:53]
	v_mfma_f32_16x16x32_f16 v[46:49], a[36:39], a[72:75], v[46:49]
	v_mfma_f32_16x16x32_f16 v[42:45], a[28:31], a[72:75], v[42:45]
	v_mfma_f32_16x16x32_f16 v[38:41], a[12:15], a[72:75], v[38:41]
	v_mfma_f32_16x16x32_f16 v[34:37], a[20:23], a[72:75], v[34:37]
	v_mfma_f32_16x16x32_f16 v[30:33], a[68:71], a[4:7], v[30:33]
	v_mfma_f32_16x16x32_f16 v[26:29], a[60:63], a[4:7], v[26:29]
	v_mfma_f32_16x16x32_f16 v[22:25], a[52:55], a[4:7], v[22:25]
	v_mfma_f32_16x16x32_f16 v[18:21], a[44:47], a[4:7], v[18:21]
	v_mfma_f32_16x16x32_f16 v[14:17], a[36:39], a[4:7], v[14:17]
	v_mfma_f32_16x16x32_f16 v[10:13], a[28:31], a[4:7], v[10:13]
	v_mfma_f32_16x16x32_f16 a[134:137], a[12:15], a[4:7], a[134:137]
	v_mfma_f32_16x16x32_f16 a[130:133], a[20:23], a[4:7], a[130:133]
	s_cbranch_scc0 .LBB0_5
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_cmp_gt_i32_e32 vcc, s10, v254
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_2
; %bb.4:                                ; %.critedge
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_mov_b32 m0, s20
	s_and_b32 s1, s19, 0xffff
	s_mov_b32 s0, s11
	v_accvgpr_read_b32 v6, a138
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s21
	v_accvgpr_read_b32 v6, a139
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	v_accvgpr_read_b32 v6, a140
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s23
	v_accvgpr_read_b32 v6, a141
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	v_accvgpr_read_b32 v6, a142
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s25
	v_accvgpr_read_b32 v6, a143
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	v_accvgpr_read_b32 v6, a144
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s27
	v_accvgpr_read_b32 v6, a145
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_and_b32 s1, s18, 0xffff
	s_mov_b32 s0, s12
	s_mov_b32 m0, s28
	v_accvgpr_read_b32 v6, a146
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s29
	v_accvgpr_read_b32 v6, a147
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	v_accvgpr_read_b32 v6, a148
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s31
	v_accvgpr_read_b32 v6, a149
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	v_accvgpr_read_b32 v6, a150
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s34
	v_accvgpr_read_b32 v6, a151
	buffer_load_dwordx4 v6, s[0:3], 0 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v255, s[0:3], 0 offen lds
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v0, s[0:3], 0 offen lds
	s_branch .LBB0_2
.LBB0_5:                                ; %Flow
	v_accvgpr_read_b32 v0, a128
.LBB0_6:                                ; %._crit_edge
	v_and_b32_e32 v0, 15, v0
	s_lshl_b32 s0, s15, 3
	v_and_or_b32 v0, s0, 16, v0
	s_mul_i32 s0, s16, s13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_cvt_pk_f16_f32 v18, v18, v19
	v_cvt_pk_f16_f32 v19, v20, v21
	v_accvgpr_read_b32 v20, a129
	s_add_u32 s2, s6, s0
	v_accvgpr_read_b32 v6, a130
	v_lshrrev_b32_e32 v20, 2, v20
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s15, s14, 31
	v_accvgpr_read_b32 v7, a131
	v_accvgpr_read_b32 v8, a132
	v_accvgpr_read_b32 v9, a133
	v_and_b32_e32 v20, 28, v20
	s_lshl_b64 s[0:1], s[14:15], 1
	v_cvt_pk_f16_f32 v34, v34, v35
	v_cvt_pk_f16_f32 v35, v36, v37
	v_cvt_pk_f16_f32 v30, v30, v31
	v_cvt_pk_f16_f32 v31, v32, v33
	v_cvt_pk_f16_f32 v26, v26, v27
	v_cvt_pk_f16_f32 v27, v28, v29
	v_cvt_pk_f16_f32 v22, v22, v23
	v_cvt_pk_f16_f32 v23, v24, v25
	v_cvt_pk_f16_f32 v14, v14, v15
	v_cvt_pk_f16_f32 v15, v16, v17
	v_cvt_pk_f16_f32 v10, v10, v11
	v_cvt_pk_f16_f32 v11, v12, v13
	v_cvt_pk_f16_f32 v6, v6, v7
	v_cvt_pk_f16_f32 v7, v8, v9
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
	v_or_b32_e32 v28, 0x80, v20
	v_or_b32_e32 v29, 0xa0, v20
	v_or_b32_e32 v32, 0xc0, v20
	v_or_b32_e32 v33, 0xe0, v20
	s_add_u32 s36, s2, s0
	v_mul_lo_u32 v36, v0, s13
	v_cmp_gt_i32_e64 s[28:29], s8, v0
	v_cmp_gt_i32_e64 s[14:15], s9, v20
	v_cvt_pk_f16_f32 v46, v46, v47
	v_cvt_pk_f16_f32 v47, v48, v49
	v_cvt_pk_f16_f32 v42, v42, v43
	v_cvt_pk_f16_f32 v43, v44, v45
	v_cvt_pk_f16_f32 v38, v38, v39
	v_cvt_pk_f16_f32 v39, v40, v41
	s_addc_u32 s33, s3, s1
	v_mul_lo_u32 v37, v1, s13
	v_mul_lo_u32 v40, v8, s13
	v_mul_lo_u32 v41, v9, s13
	v_mul_lo_u32 v44, v12, s13
	v_mul_lo_u32 v45, v13, s13
	v_mul_lo_u32 v48, v16, s13
	v_mul_lo_u32 v49, v17, s13
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
	v_cmp_gt_i32_e64 s[6:7], s9, v28
	v_cmp_gt_i32_e64 s[4:5], s9, v29
	v_cmp_gt_i32_e64 s[2:3], s9, v32
	v_cmp_gt_i32_e64 s[0:1], s9, v33
	v_add_lshl_u32 v0, v20, v36, 1
	v_bfrev_b32_e32 v1, 1
	s_and_b64 s[8:9], s[28:29], s[14:15]
	v_cvt_pk_f16_f32 v2, v2, v3
	v_cvt_pk_f16_f32 v3, v4, v5
	s_and_b32 s37, s33, 0xffff
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[2:3], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v36, 1
	s_and_b64 s[8:9], s[28:29], s[12:13]
	v_cvt_pk_f16_f32 v254, v250, v251
	v_cvt_pk_f16_f32 v255, v252, v253
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[254:255], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v36, 1
	s_and_b64 s[8:9], s[28:29], s[10:11]
	v_cvt_pk_f16_f32 v250, v246, v247
	v_cvt_pk_f16_f32 v251, v248, v249
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[250:251], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v36, 1
	s_and_b64 s[8:9], s[28:29], s[30:31]
	v_cvt_pk_f16_f32 v246, v242, v243
	v_cvt_pk_f16_f32 v247, v244, v245
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[246:247], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v28, v36, 1
	s_and_b64 s[8:9], s[28:29], s[6:7]
	v_cvt_pk_f16_f32 v242, v238, v239
	v_cvt_pk_f16_f32 v243, v240, v241
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[242:243], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v29, v36, 1
	s_and_b64 s[8:9], s[28:29], s[4:5]
	v_cvt_pk_f16_f32 v238, v234, v235
	v_cvt_pk_f16_f32 v239, v236, v237
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[238:239], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v32, v36, 1
	s_and_b64 s[8:9], s[28:29], s[2:3]
	v_cvt_pk_f16_f32 v234, v230, v231
	v_cvt_pk_f16_f32 v235, v232, v233
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[234:235], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v33, v36, 1
	s_and_b64 s[8:9], s[28:29], s[0:1]
	v_cvt_pk_f16_f32 v230, v226, v227
	v_cvt_pk_f16_f32 v231, v228, v229
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[230:231], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v37, v20, 1
	s_and_b64 s[8:9], s[26:27], s[14:15]
	v_cvt_pk_f16_f32 v226, v222, v223
	v_cvt_pk_f16_f32 v227, v224, v225
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[226:227], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v37, 1
	s_and_b64 s[8:9], s[26:27], s[12:13]
	v_cvt_pk_f16_f32 v222, v218, v219
	v_cvt_pk_f16_f32 v223, v220, v221
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[222:223], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v37, 1
	s_and_b64 s[8:9], s[26:27], s[10:11]
	v_cvt_pk_f16_f32 v218, v214, v215
	v_cvt_pk_f16_f32 v219, v216, v217
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[218:219], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v37, 1
	s_and_b64 s[8:9], s[26:27], s[30:31]
	v_cvt_pk_f16_f32 v214, v210, v211
	v_cvt_pk_f16_f32 v215, v212, v213
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[214:215], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v28, v37, 1
	s_and_b64 s[8:9], s[26:27], s[6:7]
	v_cvt_pk_f16_f32 v210, v206, v207
	v_cvt_pk_f16_f32 v211, v208, v209
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[210:211], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v29, v37, 1
	s_and_b64 s[8:9], s[26:27], s[4:5]
	v_cvt_pk_f16_f32 v206, v198, v199
	v_cvt_pk_f16_f32 v207, v200, v201
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[206:207], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v32, v37, 1
	s_and_b64 s[8:9], s[26:27], s[2:3]
	v_cvt_pk_f16_f32 v198, v202, v203
	v_cvt_pk_f16_f32 v199, v204, v205
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[198:199], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v33, v37, 1
	s_and_b64 s[8:9], s[26:27], s[0:1]
	v_cvt_pk_f16_f32 v194, v194, v195
	v_cvt_pk_f16_f32 v195, v196, v197
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[194:195], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v40, v20, 1
	s_and_b64 s[8:9], s[24:25], s[14:15]
	v_cvt_pk_f16_f32 v190, v190, v191
	v_cvt_pk_f16_f32 v191, v192, v193
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[190:191], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v40, 1
	s_and_b64 s[8:9], s[24:25], s[12:13]
	v_cvt_pk_f16_f32 v186, v186, v187
	v_cvt_pk_f16_f32 v187, v188, v189
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[186:187], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v40, 1
	s_and_b64 s[8:9], s[24:25], s[10:11]
	v_cvt_pk_f16_f32 v182, v182, v183
	v_cvt_pk_f16_f32 v183, v184, v185
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[182:183], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v40, 1
	s_and_b64 s[8:9], s[24:25], s[30:31]
	v_cvt_pk_f16_f32 v178, v178, v179
	v_cvt_pk_f16_f32 v179, v180, v181
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[178:179], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v28, v40, 1
	s_and_b64 s[8:9], s[24:25], s[6:7]
	v_cvt_pk_f16_f32 v174, v174, v175
	v_cvt_pk_f16_f32 v175, v176, v177
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[174:175], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v29, v40, 1
	s_and_b64 s[8:9], s[24:25], s[4:5]
	v_cvt_pk_f16_f32 v170, v170, v171
	v_cvt_pk_f16_f32 v171, v172, v173
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[170:171], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v32, v40, 1
	s_and_b64 s[8:9], s[24:25], s[2:3]
	v_cvt_pk_f16_f32 v166, v166, v167
	v_cvt_pk_f16_f32 v167, v168, v169
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[166:167], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v33, v40, 1
	s_and_b64 s[8:9], s[24:25], s[0:1]
	v_cvt_pk_f16_f32 v162, v162, v163
	v_cvt_pk_f16_f32 v163, v164, v165
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[162:163], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v41, v20, 1
	s_and_b64 s[8:9], s[22:23], s[14:15]
	v_cvt_pk_f16_f32 v158, v158, v159
	v_cvt_pk_f16_f32 v159, v160, v161
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[158:159], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v41, 1
	s_and_b64 s[8:9], s[22:23], s[12:13]
	v_cvt_pk_f16_f32 v154, v154, v155
	v_cvt_pk_f16_f32 v155, v156, v157
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[154:155], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v41, 1
	s_and_b64 s[8:9], s[22:23], s[10:11]
	v_cvt_pk_f16_f32 v150, v150, v151
	v_cvt_pk_f16_f32 v151, v152, v153
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[150:151], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v41, 1
	s_and_b64 s[8:9], s[22:23], s[30:31]
	v_cvt_pk_f16_f32 v146, v146, v147
	v_cvt_pk_f16_f32 v147, v148, v149
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[146:147], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v28, v41, 1
	s_and_b64 s[8:9], s[22:23], s[6:7]
	v_cvt_pk_f16_f32 v142, v142, v143
	v_cvt_pk_f16_f32 v143, v144, v145
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[142:143], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v29, v41, 1
	s_and_b64 s[8:9], s[22:23], s[4:5]
	v_cvt_pk_f16_f32 v138, v138, v139
	v_cvt_pk_f16_f32 v139, v140, v141
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[138:139], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v32, v41, 1
	s_and_b64 s[8:9], s[22:23], s[2:3]
	v_cvt_pk_f16_f32 v134, v134, v135
	v_cvt_pk_f16_f32 v135, v136, v137
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[134:135], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v33, v41, 1
	s_and_b64 s[8:9], s[22:23], s[0:1]
	v_cvt_pk_f16_f32 v130, v130, v131
	v_cvt_pk_f16_f32 v131, v132, v133
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[130:131], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v44, v20, 1
	s_and_b64 s[8:9], s[20:21], s[14:15]
	v_cvt_pk_f16_f32 v126, v126, v127
	v_cvt_pk_f16_f32 v127, v128, v129
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[126:127], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v44, 1
	s_and_b64 s[8:9], s[20:21], s[12:13]
	v_cvt_pk_f16_f32 v122, v122, v123
	v_cvt_pk_f16_f32 v123, v124, v125
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[122:123], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v44, 1
	s_and_b64 s[8:9], s[20:21], s[10:11]
	v_cvt_pk_f16_f32 v118, v118, v119
	v_cvt_pk_f16_f32 v119, v120, v121
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[118:119], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v44, 1
	s_and_b64 s[8:9], s[20:21], s[30:31]
	v_cvt_pk_f16_f32 v114, v114, v115
	v_cvt_pk_f16_f32 v115, v116, v117
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[114:115], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v28, v44, 1
	s_and_b64 s[8:9], s[20:21], s[6:7]
	v_cvt_pk_f16_f32 v110, v110, v111
	v_cvt_pk_f16_f32 v111, v112, v113
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[110:111], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v29, v44, 1
	s_and_b64 s[8:9], s[20:21], s[4:5]
	v_cvt_pk_f16_f32 v106, v106, v107
	v_cvt_pk_f16_f32 v107, v108, v109
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[106:107], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v32, v44, 1
	s_and_b64 s[8:9], s[20:21], s[2:3]
	v_cvt_pk_f16_f32 v102, v102, v103
	v_cvt_pk_f16_f32 v103, v104, v105
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[102:103], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v33, v44, 1
	s_and_b64 s[8:9], s[20:21], s[0:1]
	v_cvt_pk_f16_f32 v98, v98, v99
	v_cvt_pk_f16_f32 v99, v100, v101
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[98:99], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v45, v20, 1
	s_and_b64 s[8:9], s[18:19], s[14:15]
	v_cvt_pk_f16_f32 v94, v94, v95
	v_cvt_pk_f16_f32 v95, v96, v97
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[94:95], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v45, 1
	s_and_b64 s[8:9], s[18:19], s[12:13]
	v_cvt_pk_f16_f32 v90, v90, v91
	v_cvt_pk_f16_f32 v91, v92, v93
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[90:91], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v45, 1
	s_and_b64 s[8:9], s[18:19], s[10:11]
	v_cvt_pk_f16_f32 v86, v86, v87
	v_cvt_pk_f16_f32 v87, v88, v89
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[86:87], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v45, 1
	s_and_b64 s[8:9], s[18:19], s[30:31]
	v_cvt_pk_f16_f32 v82, v82, v83
	v_cvt_pk_f16_f32 v83, v84, v85
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[82:83], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v28, v45, 1
	s_and_b64 s[8:9], s[18:19], s[6:7]
	v_cvt_pk_f16_f32 v78, v78, v79
	v_cvt_pk_f16_f32 v79, v80, v81
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[78:79], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v29, v45, 1
	s_and_b64 s[8:9], s[18:19], s[4:5]
	v_cvt_pk_f16_f32 v74, v74, v75
	v_cvt_pk_f16_f32 v75, v76, v77
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[74:75], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v32, v45, 1
	s_and_b64 s[8:9], s[18:19], s[2:3]
	v_cvt_pk_f16_f32 v70, v70, v71
	v_cvt_pk_f16_f32 v71, v72, v73
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[70:71], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v33, v45, 1
	s_and_b64 s[8:9], s[18:19], s[0:1]
	v_cvt_pk_f16_f32 v66, v66, v67
	v_cvt_pk_f16_f32 v67, v68, v69
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[66:67], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v48, v20, 1
	s_and_b64 s[8:9], s[16:17], s[14:15]
	v_cvt_pk_f16_f32 v62, v62, v63
	v_cvt_pk_f16_f32 v63, v64, v65
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[62:63], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v48, 1
	s_and_b64 s[8:9], s[16:17], s[12:13]
	v_cvt_pk_f16_f32 v58, v58, v59
	v_cvt_pk_f16_f32 v59, v60, v61
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[58:59], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v48, 1
	s_and_b64 s[8:9], s[16:17], s[10:11]
	v_cvt_pk_f16_f32 v54, v54, v55
	v_cvt_pk_f16_f32 v55, v56, v57
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[54:55], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v48, 1
	s_and_b64 s[8:9], s[16:17], s[30:31]
	v_cvt_pk_f16_f32 v50, v50, v51
	v_cvt_pk_f16_f32 v51, v52, v53
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[50:51], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v28, v48, 1
	s_and_b64 s[8:9], s[16:17], s[6:7]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[46:47], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v29, v48, 1
	s_and_b64 s[8:9], s[16:17], s[4:5]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[42:43], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v32, v48, 1
	s_and_b64 s[8:9], s[16:17], s[2:3]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[38:39], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v33, v48, 1
	s_and_b64 s[8:9], s[16:17], s[0:1]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[34:35], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v49, v20, 1
	s_and_b64 s[8:9], vcc, s[14:15]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[30:31], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v21, v49, 1
	s_and_b64 s[8:9], vcc, s[12:13]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[26:27], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v24, v49, 1
	s_and_b64 s[8:9], vcc, s[10:11]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[22:23], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v25, v49, 1
	s_and_b64 s[8:9], vcc, s[30:31]
	v_cndmask_b32_e64 v0, v1, v0, s[8:9]
	buffer_store_dwordx2 v[18:19], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v28, v49, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_cndmask_b32_e64 v0, v1, v0, s[6:7]
	buffer_store_dwordx2 v[14:15], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v29, v49, 1
	s_and_b64 s[4:5], vcc, s[4:5]
	v_accvgpr_read_b32 v203, a137
	v_cndmask_b32_e64 v0, v1, v0, s[4:5]
	v_accvgpr_read_b32 v202, a136
	v_accvgpr_read_b32 v201, a135
	v_accvgpr_read_b32 v200, a134
	buffer_store_dwordx2 v[10:11], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v32, v49, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_cvt_pk_f16_f32 v4, v200, v201
	v_cvt_pk_f16_f32 v5, v202, v203
	v_cndmask_b32_e64 v0, v1, v0, s[2:3]
	buffer_store_dwordx2 v[4:5], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v33, v49, 1
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v0, v1, v0, vcc
	buffer_store_dwordx2 v[6:7], v0, s[36:39], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v2_async_copy
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
		.amdhsa_next_free_vgpr 408
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
	.size	v2_async_copy, .Lfunc_end0-v2_async_copy
	.cfi_endproc
                                        ; -- End function
	.set v2_async_copy.num_vgpr, 256
	.set v2_async_copy.num_agpr, 152
	.set v2_async_copy.numbered_sgpr, 40
	.set v2_async_copy.num_named_barrier, 0
	.set v2_async_copy.private_seg_size, 0
	.set v2_async_copy.uses_vcc, 1
	.set v2_async_copy.uses_flat_scratch, 0
	.set v2_async_copy.has_dyn_sized_stack, 0
	.set v2_async_copy.has_recursion, 0
	.set v2_async_copy.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8264
; TotalNumSgprs: 46
; NumVgprs: 256
; NumAgprs: 152
; TotalNumVgprs: 408
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 50
; NumSGPRsForWavesPerEU: 46
; NumVGPRsForWavesPerEU: 408
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
	.byte	16                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	66                              ; DW_AT_call_line
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
	.asciz	"/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v2_async_copy" ; string offset=24
.Linfo_string3:
	.asciz	"v2_async_copy"                 ; string offset=85
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     152
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
    .name:           v2_async_copy
    .private_segment_fixed_size: 0
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         v2_async_copy.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     408
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
