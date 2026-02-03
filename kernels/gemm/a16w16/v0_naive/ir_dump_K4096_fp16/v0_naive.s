	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	v0_naive                        ; -- Begin function v0_naive
	.p2align	8
	.type	v0_naive,@function
v0_naive:                               ; @v0_naive
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.165:
	.file	1 "/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v0_naive" "matmul_kernel.py"
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.166:
.LBB0_0:
	.file	2 "/root/OAI-triton/python/triton/language" "standard.py"
	s_add_i32 s0, s9, 0xff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s1, s0, 8
	v_mov_b32_e32 v255, v0
	s_abs_i32 s18, s16
	s_xor_b32 s17, s16, s1
	s_ashr_i32 s17, s17, 31
	s_abs_i32 s15, s1
	s_sub_i32 s19, 0, s15
	v_cvt_f32_u32_e32 v0, s15
	v_readfirstlane_b32 s14, v255
	s_and_b32 s0, s14, 0xc0
	v_accvgpr_write_b32 a127, 0
	v_rcp_iflag_f32_e32 v0, v0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a124, 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a129, 0
	v_mul_lo_u32 v1, s19, v0
	v_mul_hi_u32 v1, v0, v1
	v_add_u32_e32 v0, v0, v1
	v_mul_hi_u32 v0, s18, v0
	v_mul_lo_u32 v1, v0, s15
	v_sub_u32_e32 v1, s18, v1
	v_add_u32_e32 v2, 1, v0
	v_subrev_u32_e32 v3, s15, v1
	v_cmp_le_u32_e32 vcc, s15, v1
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a135, 0
	v_cndmask_b32_e32 v0, v0, v2, vcc
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_add_u32_e32 v2, 1, v0
	v_cmp_le_u32_e32 vcc, s15, v1
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a133, 0
	v_cndmask_b32_e32 v0, v0, v2, vcc
	v_xor_b32_e32 v0, s17, v0
	v_subrev_u32_e32 v0, s17, v0
	v_mul_lo_u32 v1, v0, s1
	v_sub_u32_e32 v1, s16, v1
	v_lshlrev_b32_e32 v8, 8, v0
	s_add_i32 s1, s10, 63
	v_and_b32_e32 v0, 63, v255
	s_cmp_lt_i32 s1, 64
	v_accvgpr_write_b32 a132, 0
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
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
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
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v202, 0
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
	v_or_b32_e32 v9, s0, v0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v10, 8, v1
	v_mov_b32_e32 v2, 0
	s_cbranch_scc1 .LBB0_36
; %bb.1:                                ; %.lr.ph
	v_lshlrev_b32_e32 v6, 4, v255
	v_lshlrev_b32_e32 v7, 3, v9
	v_accvgpr_write_b32 a137, v8
	v_mul_lo_u32 v4, v8, s11
	v_and_b32_e32 v6, 0x60, v6
	v_and_b32_e32 v7, 0x300, v7
	v_and_b32_e32 v8, 0xe0, v9
	v_lshlrev_b32_e32 v11, 1, v255
	v_and_b32_e32 v12, 16, v255
	v_lshlrev_b32_e32 v1, 3, v255
	s_ashr_i32 s15, s1, 31
	v_and_b32_e32 v11, 16, v11
	v_bitop3_b32 v7, v7, v8, v6 bitop3:0x36
	v_lshl_add_u32 v8, v12, 7, 0
	v_and_b32_e32 v22, 56, v1
	s_lshr_b32 s15, s15, 26
	v_mov_b32_e32 v15, v9
	v_and_b32_e32 v9, 1, v255
	v_add3_u32 v7, v8, v7, v11
	v_lshlrev_b32_e32 v8, 6, v255
	v_and_b32_e32 v1, 0x60, v1
	s_movk_i32 s16, 0xa0
	s_add_i32 s1, s1, s15
	v_accvgpr_write_b32 a138, v10
	v_mul_lo_u32 v2, v10, s12
	v_lshlrev_b32_e32 v10, 10, v9
	v_and_b32_e32 v8, 0x700, v8
	v_lshlrev_b32_e32 v9, 4, v9
	v_bitop3_b32 v14, v15, v1, s16 bitop3:0x6c
	s_ashr_i32 s15, s1, 6
	s_movk_i32 s1, 0xe0
	v_or3_b32 v8, v8, v14, v9
	v_lshlrev_b32_e32 v14, 7, v255
	v_and_b32_e32 v14, 0xb00, v14
	v_bitop3_b32 v6, v15, v6, s1 bitop3:0x6c
	s_lshl_b32 s1, s14, 1
	v_add_u32_e32 v14, 0, v14
	s_and_b32 s1, s1, 0x80
	v_lshlrev_b32_e32 v13, 10, v255
	v_add3_u32 v6, v14, v6, v10
	v_bfe_i32 v14, v255, 5, 1
	v_lshl_or_b32 v12, v12, 6, s1
	s_movk_i32 s1, 0x120
	v_and_b32_e32 v13, 0x800, v13
	v_bitop3_b32 v1, v14, v1, s1 bitop3:0x6c
	v_add_u32_e32 v0, s0, v0
	v_or_b32_e32 v254, v8, v13
	v_bitop3_b32 v8, v8, 64, v13 bitop3:0x36
	v_or3_b32 v1, v12, v1, v13
	v_lshrrev_b32_e32 v13, 3, v0
	v_ashrrev_i32_e32 v3, 31, v2
	v_add_u32_e32 v14, 0xe0, v13
	v_or_b32_e32 v12, v1, v9
	v_bitop3_b32 v9, v1, 64, v9 bitop3:0x36
	v_lshlrev_b64 v[2:3], 1, v[2:3]
	v_mad_u64_u32 v[0:1], s[0:1], s12, v14, v[22:23]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[2:3]
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[2:3]
	v_accvgpr_write_b32 a139, v15
	v_accvgpr_write_b32 a143, v1
	v_or_b32_e32 v15, 0xc0, v13
	v_accvgpr_write_b32 a142, v0
	v_mad_u64_u32 v[0:1], s[0:1], s12, v15, v[22:23]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[2:3]
	v_accvgpr_write_b32 a145, v1
	v_add_u32_e32 v16, 0xa0, v13
	v_accvgpr_write_b32 a144, v0
	v_mad_u64_u32 v[0:1], s[0:1], s12, v16, v[22:23]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[2:3]
	v_accvgpr_write_b32 a147, v1
	v_or_b32_e32 v17, 0x80, v13
	v_accvgpr_write_b32 a146, v0
	v_mad_u64_u32 v[0:1], s[0:1], s12, v17, v[22:23]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[2:3]
	v_accvgpr_write_b32 a149, v1
	v_add_u32_e32 v18, 0x60, v13
	v_accvgpr_write_b32 a148, v0
	v_mad_u64_u32 v[0:1], s[0:1], s12, v18, v[22:23]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[2:3]
	v_accvgpr_write_b32 a151, v1
	v_or_b32_e32 v19, 64, v13
	v_accvgpr_write_b32 a150, v0
	v_mad_u64_u32 v[0:1], s[0:1], s12, v19, v[22:23]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[2:3]
	v_accvgpr_write_b32 a153, v1
	v_add_u32_e32 v20, 32, v13
	v_accvgpr_write_b32 a152, v0
	v_mad_u64_u32 v[0:1], s[0:1], s12, v20, v[22:23]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[2:3]
	v_accvgpr_write_b32 a155, v1
	v_accvgpr_write_b32 a154, v0
	v_mad_u64_u32 v[0:1], s[0:1], s12, v13, v[22:23]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[2:3]
	v_ashrrev_i32_e32 v5, 31, v4
	v_accvgpr_write_b32 a157, v1
	v_accvgpr_write_b32 a156, v0
	v_lshlrev_b64 v[0:1], 1, v[4:5]
	v_mad_u64_u32 v[2:3], s[0:1], s11, v14, v[22:23]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	v_accvgpr_write_b32 a159, v3
	v_accvgpr_write_b32 a158, v2
	v_mad_u64_u32 v[2:3], s[0:1], s11, v15, v[22:23]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	v_accvgpr_write_b32 a161, v3
	v_accvgpr_write_b32 a160, v2
	v_mad_u64_u32 v[2:3], s[0:1], s11, v16, v[22:23]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	v_accvgpr_write_b32 a163, v3
	v_accvgpr_write_b32 a162, v2
	v_mad_u64_u32 v[2:3], s[0:1], s11, v17, v[22:23]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	v_accvgpr_write_b32 a165, v3
	v_accvgpr_write_b32 a164, v2
	v_mad_u64_u32 v[2:3], s[0:1], s11, v18, v[22:23]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	v_accvgpr_write_b32 a167, v3
	v_accvgpr_write_b32 a166, v2
	v_mad_u64_u32 v[2:3], s[0:1], s11, v19, v[22:23]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	v_accvgpr_write_b32 a169, v3
	v_accvgpr_write_b32 a168, v2
	v_mad_u64_u32 v[2:3], s[0:1], s11, v20, v[22:23]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	v_accvgpr_write_b32 a171, v3
	v_accvgpr_write_b32 a170, v2
	v_mad_u64_u32 v[2:3], s[0:1], s11, v13, v[22:23]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[0:1], v[2:3], 1, v[0:1]
	v_accvgpr_write_b32 a173, v1
	v_accvgpr_write_b32 a172, v0
	v_accvgpr_write_b32 a136, v255
	v_accvgpr_write_b32 a140, v22
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v2, 0
	v_add_u32_e32 v1, 0, v254
	v_add_u32_e32 v254, 0, v8
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v3, 0
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
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
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
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_add_u32_e32 v255, v7, v10
	v_add_u32_e32 v0, v6, v11
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
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
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_add_u32_e32 v6, 0, v12
	v_add_u32_e32 v7, 0, v9
	v_accvgpr_write_b32 a127, v8
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b128 v255, a[0:3]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[10:13], v1
	ds_read_b128 a[0:3], v254
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v255, a[8:11]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[8:11], v1
	ds_read_b128 a[64:67], v254
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v255, a[4:7]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[4:7], v1
	ds_read_b128 a[68:71], v254
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v255, a[16:19]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[16:19], v1
	ds_read_b128 a[72:75], v254
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v255, a[12:15]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[12:15], v1
	ds_read_b128 a[76:79], v254
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v255, a[24:27]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[24:27], v1
	ds_read_b128 a[80:83], v254
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v255, a[20:23]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[20:23], v1
	ds_read_b128 a[84:87], v254
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v255, a[36:39]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[36:39], v1
	ds_read_b128 a[88:91], v254
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, a[32:35]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[32:35], v6
	ds_read_b128 a[92:95], v7 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, a[48:51]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[48:51], v6
	ds_read_b128 a[96:99], v7 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, a[28:31]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[28:31], v6
	ds_read_b128 a[100:103], v7 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, a[44:47]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[44:47], v6
	ds_read_b128 a[104:107], v7 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, a[40:43]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[40:43], v6
	ds_read_b128 a[108:111], v7 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, a[56:59]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[56:59], v6
	ds_read_b128 a[112:115], v7 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, a[52:55]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[52:55], v6
	ds_read_b128 a[116:119], v7 offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, a[60:63]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[60:63], v6
	ds_read_b128 a[120:123], v7 offset:512
	v_mfma_f32_16x16x32_f16 v[2:5], a[32:35], v[10:13], v[2:5]
	s_add_u32 s0, s0, 0x80
	s_addc_u32 s1, s1, 0
	s_add_i32 s15, s15, -1
	v_mfma_f32_16x16x32_f16 v[250:253], a[48:51], v[10:13], v[250:253]
	s_sub_i32 s10, s10, 64
	s_cmp_lg_u32 s15, 0
	v_mfma_f32_16x16x32_f16 v[246:249], a[28:31], v[10:13], v[246:249]
	v_mfma_f32_16x16x32_f16 v[242:245], a[44:47], v[10:13], v[242:245]
	v_mfma_f32_16x16x32_f16 v[238:241], a[40:43], v[10:13], v[238:241]
	v_mfma_f32_16x16x32_f16 v[234:237], a[56:59], v[10:13], v[234:237]
	v_mfma_f32_16x16x32_f16 v[230:233], a[52:55], v[10:13], v[230:233]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_f16 v[226:229], a[60:63], v[10:13], v[226:229]
	v_mfma_f32_16x16x32_f16 v[222:225], a[32:35], a[8:11], v[222:225]
	v_mfma_f32_16x16x32_f16 v[218:221], a[48:51], a[8:11], v[218:221]
	v_mfma_f32_16x16x32_f16 v[214:217], a[28:31], a[8:11], v[214:217]
	v_mfma_f32_16x16x32_f16 v[210:213], a[44:47], a[8:11], v[210:213]
	v_mfma_f32_16x16x32_f16 v[206:209], a[40:43], a[8:11], v[206:209]
	v_mfma_f32_16x16x32_f16 v[202:205], a[56:59], a[8:11], v[202:205]
	v_mfma_f32_16x16x32_f16 v[198:201], a[52:55], a[8:11], v[198:201]
	v_mfma_f32_16x16x32_f16 v[194:197], a[60:63], a[8:11], v[194:197]
	v_mfma_f32_16x16x32_f16 v[190:193], a[32:35], a[4:7], v[190:193]
	v_mfma_f32_16x16x32_f16 v[186:189], a[48:51], a[4:7], v[186:189]
	v_mfma_f32_16x16x32_f16 v[182:185], a[28:31], a[4:7], v[182:185]
	v_mfma_f32_16x16x32_f16 v[178:181], a[44:47], a[4:7], v[178:181]
	v_mfma_f32_16x16x32_f16 v[174:177], a[40:43], a[4:7], v[174:177]
	v_mfma_f32_16x16x32_f16 v[170:173], a[56:59], a[4:7], v[170:173]
	v_mfma_f32_16x16x32_f16 v[166:169], a[52:55], a[4:7], v[166:169]
	v_mfma_f32_16x16x32_f16 v[162:165], a[60:63], a[4:7], v[162:165]
	v_mfma_f32_16x16x32_f16 v[158:161], a[32:35], a[16:19], v[158:161]
	v_mfma_f32_16x16x32_f16 v[154:157], a[48:51], a[16:19], v[154:157]
	v_mfma_f32_16x16x32_f16 v[150:153], a[28:31], a[16:19], v[150:153]
	v_mfma_f32_16x16x32_f16 v[146:149], a[44:47], a[16:19], v[146:149]
	v_mfma_f32_16x16x32_f16 v[142:145], a[40:43], a[16:19], v[142:145]
	v_mfma_f32_16x16x32_f16 v[138:141], a[56:59], a[16:19], v[138:141]
	v_mfma_f32_16x16x32_f16 v[134:137], a[52:55], a[16:19], v[134:137]
	v_mfma_f32_16x16x32_f16 v[130:133], a[60:63], a[16:19], v[130:133]
	v_mfma_f32_16x16x32_f16 v[126:129], a[32:35], a[12:15], v[126:129]
	v_mfma_f32_16x16x32_f16 v[122:125], a[48:51], a[12:15], v[122:125]
	v_mfma_f32_16x16x32_f16 v[118:121], a[28:31], a[12:15], v[118:121]
	v_mfma_f32_16x16x32_f16 v[114:117], a[44:47], a[12:15], v[114:117]
	v_mfma_f32_16x16x32_f16 v[110:113], a[40:43], a[12:15], v[110:113]
	v_mfma_f32_16x16x32_f16 v[106:109], a[56:59], a[12:15], v[106:109]
	v_mfma_f32_16x16x32_f16 v[102:105], a[52:55], a[12:15], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[60:63], a[12:15], v[98:101]
	v_mfma_f32_16x16x32_f16 v[94:97], a[32:35], a[24:27], v[94:97]
	v_mfma_f32_16x16x32_f16 v[90:93], a[48:51], a[24:27], v[90:93]
	v_mfma_f32_16x16x32_f16 v[86:89], a[28:31], a[24:27], v[86:89]
	v_mfma_f32_16x16x32_f16 v[82:85], a[44:47], a[24:27], v[82:85]
	v_mfma_f32_16x16x32_f16 v[78:81], a[40:43], a[24:27], v[78:81]
	v_mfma_f32_16x16x32_f16 v[74:77], a[56:59], a[24:27], v[74:77]
	v_mfma_f32_16x16x32_f16 v[70:73], a[52:55], a[24:27], v[70:73]
	v_mfma_f32_16x16x32_f16 v[66:69], a[60:63], a[24:27], v[66:69]
	v_mfma_f32_16x16x32_f16 v[58:61], a[32:35], a[20:23], v[58:61]
	v_mfma_f32_16x16x32_f16 v[50:53], a[48:51], a[20:23], v[50:53]
	v_mfma_f32_16x16x32_f16 v[42:45], a[28:31], a[20:23], v[42:45]
	v_mfma_f32_16x16x32_f16 v[62:65], a[44:47], a[20:23], v[62:65]
	v_mfma_f32_16x16x32_f16 v[54:57], a[40:43], a[20:23], v[54:57]
	v_mfma_f32_16x16x32_f16 v[46:49], a[56:59], a[20:23], v[46:49]
	v_mfma_f32_16x16x32_f16 v[38:41], a[52:55], a[20:23], v[38:41]
	v_mfma_f32_16x16x32_f16 v[34:37], a[60:63], a[20:23], v[34:37]
	v_mfma_f32_16x16x32_f16 v[30:33], a[32:35], a[36:39], v[30:33]
	v_mfma_f32_16x16x32_f16 v[26:29], a[48:51], a[36:39], v[26:29]
	v_mfma_f32_16x16x32_f16 v[22:25], a[28:31], a[36:39], v[22:25]
	v_mfma_f32_16x16x32_f16 v[18:21], a[44:47], a[36:39], v[18:21]
	v_mfma_f32_16x16x32_f16 v[14:17], a[40:43], a[36:39], v[14:17]
	v_mfma_f32_16x16x32_f16 a[132:135], a[56:59], a[36:39], a[132:135]
	v_mfma_f32_16x16x32_f16 a[128:131], a[52:55], a[36:39], a[128:131]
	v_mfma_f32_16x16x32_f16 a[124:127], a[60:63], a[36:39], a[124:127]
	v_mfma_f32_16x16x32_f16 v[2:5], a[92:95], a[0:3], v[2:5]
	v_mfma_f32_16x16x32_f16 v[250:253], a[96:99], a[0:3], v[250:253]
	v_mfma_f32_16x16x32_f16 v[246:249], a[100:103], a[0:3], v[246:249]
	v_mfma_f32_16x16x32_f16 v[242:245], a[104:107], a[0:3], v[242:245]
	v_mfma_f32_16x16x32_f16 v[238:241], a[108:111], a[0:3], v[238:241]
	v_mfma_f32_16x16x32_f16 v[234:237], a[112:115], a[0:3], v[234:237]
	v_mfma_f32_16x16x32_f16 v[230:233], a[116:119], a[0:3], v[230:233]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 v[226:229], a[120:123], a[0:3], v[226:229]
	v_mfma_f32_16x16x32_f16 v[222:225], a[92:95], a[64:67], v[222:225]
	v_mfma_f32_16x16x32_f16 v[218:221], a[96:99], a[64:67], v[218:221]
	v_mfma_f32_16x16x32_f16 v[214:217], a[100:103], a[64:67], v[214:217]
	v_mfma_f32_16x16x32_f16 v[210:213], a[104:107], a[64:67], v[210:213]
	v_mfma_f32_16x16x32_f16 v[206:209], a[108:111], a[64:67], v[206:209]
	v_mfma_f32_16x16x32_f16 v[202:205], a[112:115], a[64:67], v[202:205]
	v_mfma_f32_16x16x32_f16 v[198:201], a[116:119], a[64:67], v[198:201]
	v_mfma_f32_16x16x32_f16 v[194:197], a[120:123], a[64:67], v[194:197]
	v_mfma_f32_16x16x32_f16 v[190:193], a[92:95], a[68:71], v[190:193]
	v_mfma_f32_16x16x32_f16 v[186:189], a[96:99], a[68:71], v[186:189]
	v_mfma_f32_16x16x32_f16 v[182:185], a[100:103], a[68:71], v[182:185]
	v_mfma_f32_16x16x32_f16 v[178:181], a[104:107], a[68:71], v[178:181]
	v_mfma_f32_16x16x32_f16 v[174:177], a[108:111], a[68:71], v[174:177]
	v_mfma_f32_16x16x32_f16 v[170:173], a[112:115], a[68:71], v[170:173]
	v_mfma_f32_16x16x32_f16 v[166:169], a[116:119], a[68:71], v[166:169]
	v_mfma_f32_16x16x32_f16 v[162:165], a[120:123], a[68:71], v[162:165]
	v_mfma_f32_16x16x32_f16 v[158:161], a[92:95], a[72:75], v[158:161]
	v_mfma_f32_16x16x32_f16 v[154:157], a[96:99], a[72:75], v[154:157]
	v_mfma_f32_16x16x32_f16 v[150:153], a[100:103], a[72:75], v[150:153]
	v_mfma_f32_16x16x32_f16 v[146:149], a[104:107], a[72:75], v[146:149]
	v_mfma_f32_16x16x32_f16 v[142:145], a[108:111], a[72:75], v[142:145]
	v_mfma_f32_16x16x32_f16 v[138:141], a[112:115], a[72:75], v[138:141]
	v_mfma_f32_16x16x32_f16 v[134:137], a[116:119], a[72:75], v[134:137]
	v_mfma_f32_16x16x32_f16 v[130:133], a[120:123], a[72:75], v[130:133]
	v_mfma_f32_16x16x32_f16 v[126:129], a[92:95], a[76:79], v[126:129]
	v_mfma_f32_16x16x32_f16 v[122:125], a[96:99], a[76:79], v[122:125]
	v_mfma_f32_16x16x32_f16 v[118:121], a[100:103], a[76:79], v[118:121]
	v_mfma_f32_16x16x32_f16 v[114:117], a[104:107], a[76:79], v[114:117]
	v_mfma_f32_16x16x32_f16 v[110:113], a[108:111], a[76:79], v[110:113]
	v_mfma_f32_16x16x32_f16 v[106:109], a[112:115], a[76:79], v[106:109]
	v_mfma_f32_16x16x32_f16 v[102:105], a[116:119], a[76:79], v[102:105]
	v_mfma_f32_16x16x32_f16 v[98:101], a[120:123], a[76:79], v[98:101]
	v_mfma_f32_16x16x32_f16 v[94:97], a[92:95], a[80:83], v[94:97]
	v_mfma_f32_16x16x32_f16 v[90:93], a[96:99], a[80:83], v[90:93]
	v_mfma_f32_16x16x32_f16 v[86:89], a[100:103], a[80:83], v[86:89]
	v_mfma_f32_16x16x32_f16 v[82:85], a[104:107], a[80:83], v[82:85]
	v_mfma_f32_16x16x32_f16 v[78:81], a[108:111], a[80:83], v[78:81]
	v_mfma_f32_16x16x32_f16 v[74:77], a[112:115], a[80:83], v[74:77]
	v_mfma_f32_16x16x32_f16 v[70:73], a[116:119], a[80:83], v[70:73]
	v_mfma_f32_16x16x32_f16 v[66:69], a[120:123], a[80:83], v[66:69]
	v_mfma_f32_16x16x32_f16 v[58:61], a[92:95], a[84:87], v[58:61]
	v_mfma_f32_16x16x32_f16 v[50:53], a[96:99], a[84:87], v[50:53]
	v_mfma_f32_16x16x32_f16 v[42:45], a[100:103], a[84:87], v[42:45]
	v_mfma_f32_16x16x32_f16 v[62:65], a[104:107], a[84:87], v[62:65]
	v_mfma_f32_16x16x32_f16 v[54:57], a[108:111], a[84:87], v[54:57]
	v_mfma_f32_16x16x32_f16 v[46:49], a[112:115], a[84:87], v[46:49]
	v_mfma_f32_16x16x32_f16 v[38:41], a[116:119], a[84:87], v[38:41]
	v_mfma_f32_16x16x32_f16 v[34:37], a[120:123], a[84:87], v[34:37]
	v_mfma_f32_16x16x32_f16 v[30:33], a[92:95], a[88:91], v[30:33]
	v_mfma_f32_16x16x32_f16 v[26:29], a[96:99], a[88:91], v[26:29]
	v_mfma_f32_16x16x32_f16 v[22:25], a[100:103], a[88:91], v[22:25]
	v_mfma_f32_16x16x32_f16 v[18:21], a[104:107], a[88:91], v[18:21]
	v_mfma_f32_16x16x32_f16 v[14:17], a[108:111], a[88:91], v[14:17]
	v_mfma_f32_16x16x32_f16 a[132:135], a[112:115], a[88:91], a[132:135]
	v_mfma_f32_16x16x32_f16 a[128:131], a[116:119], a[88:91], a[128:131]
	v_mfma_f32_16x16x32_f16 a[124:127], a[120:123], a[88:91], a[124:127]
	s_cbranch_scc0 .LBB0_35
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_accvgpr_read_b32 v10, a140
	v_cmp_gt_i32_e32 vcc, s10, v10
	v_accvgpr_write_b32 a0, v8
	v_accvgpr_write_b32 a1, v8
	v_accvgpr_write_b32 a2, v8
	v_accvgpr_write_b32 a3, v8
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_5
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a172
	v_accvgpr_read_b32 v11, a173
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[0:3], v[10:11], off
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_mov_b32 a11, a4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a170
	v_accvgpr_read_b32 v11, a171
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[8:11], v[10:11], off
.LBB0_7:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_mov_b32 a5, a4
	v_accvgpr_mov_b32 a6, a4
	v_accvgpr_mov_b32 a7, a4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_9
; %bb.8:                                ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a168
	v_accvgpr_read_b32 v11, a169
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[4:7], v[10:11], off
.LBB0_9:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_mov_b32 a19, a12
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_11
; %bb.10:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a166
	v_accvgpr_read_b32 v11, a167
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[16:19], v[10:11], off
.LBB0_11:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_mov_b32 a13, a12
	v_accvgpr_mov_b32 a14, a12
	v_accvgpr_mov_b32 a15, a12
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a164
	v_accvgpr_read_b32 v11, a165
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[12:15], v[10:11], off
.LBB0_13:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_mov_b32 a27, a20
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_15
; %bb.14:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a162
	v_accvgpr_read_b32 v11, a163
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[24:27], v[10:11], off
.LBB0_15:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_mov_b32 a21, a20
	v_accvgpr_mov_b32 a22, a20
	v_accvgpr_mov_b32 a23, a20
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_17
; %bb.16:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a160
	v_accvgpr_read_b32 v11, a161
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[20:23], v[10:11], off
.LBB0_17:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_mov_b32 a39, a32
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a158
	v_accvgpr_read_b32 v11, a159
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[36:39], v[10:11], off
.LBB0_19:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_mov_b32 a33, a32
	v_accvgpr_mov_b32 a34, a32
	v_accvgpr_mov_b32 a35, a32
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a156
	v_accvgpr_read_b32 v11, a157
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[32:35], v[10:11], off
.LBB0_21:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_mov_b32 a51, a28
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a154
	v_accvgpr_read_b32 v11, a155
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[48:51], v[10:11], off
.LBB0_23:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_mov_b32 a29, a28
	v_accvgpr_mov_b32 a30, a28
	v_accvgpr_mov_b32 a31, a28
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a152
	v_accvgpr_read_b32 v11, a153
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[28:31], v[10:11], off
.LBB0_25:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_mov_b32 a47, a40
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_27
; %bb.26:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a150
	v_accvgpr_read_b32 v11, a151
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[44:47], v[10:11], off
.LBB0_27:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_mov_b32 a41, a40
	v_accvgpr_mov_b32 a42, a40
	v_accvgpr_mov_b32 a43, a40
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_29
; %bb.28:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a148
	v_accvgpr_read_b32 v11, a149
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[40:43], v[10:11], off
.LBB0_29:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_mov_b32 a59, a52
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a146
	v_accvgpr_read_b32 v11, a147
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[56:59], v[10:11], off
.LBB0_31:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_mov_b32 a53, a52
	v_accvgpr_mov_b32 a54, a52
	v_accvgpr_mov_b32 a55, a52
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a144
	v_accvgpr_read_b32 v11, a145
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[52:55], v[10:11], off
.LBB0_33:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.34:                               ;   in Loop: Header=BB0_3 Depth=1
	v_accvgpr_read_b32 v10, a142
	v_accvgpr_read_b32 v11, a143
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	global_load_dwordx4 a[60:63], v[10:11], off
	s_branch .LBB0_2
.LBB0_35:                               ; %Flow
	v_accvgpr_read_b32 v255, a136
	v_accvgpr_read_b32 v8, a137
	v_accvgpr_read_b32 v10, a138
	v_accvgpr_read_b32 v9, a139
.LBB0_36:                               ; %._crit_edge
	v_and_b32_e32 v0, 15, v255
	s_lshr_b32 s0, s14, 3
	v_and_or_b32 v7, s0, 16, v0
	v_lshrrev_b32_e32 v0, 2, v9
	v_and_b32_e32 v6, 28, v0
	v_mul_lo_u32 v0, v8, s13
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[0:1]
	v_lshlrev_b64 v[8:9], 1, v[10:11]
	v_cmp_gt_i32_e64 s[14:15], s8, v7
	v_cmp_gt_i32_e32 vcc, s9, v6
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[8:9]
	v_mul_lo_u32 v254, v7, s13
	s_and_b64 s[2:3], s[14:15], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_cvt_pk_f16_f32 v5, v4, v5
	v_cvt_pk_f16_f32 v4, v2, v3
	v_add_u32_e32 v2, v254, v6
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_dwordx2 v[2:3], v[4:5], off
.LBB0_38:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 32, v6
	v_cmp_gt_i32_e64 s[0:1], s9, v2
	s_and_b64 s[4:5], s[14:15], s[0:1]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_add_u32_e32 v8, v2, v254
	v_ashrrev_i32_e32 v9, 31, v8
	v_cvt_pk_f16_f32 v4, v250, v251
	v_cvt_pk_f16_f32 v5, v252, v253
	v_lshl_add_u64 v[8:9], v[8:9], 1, v[0:1]
	global_store_dwordx2 v[8:9], v[4:5], off
.LBB0_40:
	s_or_b64 exec, exec, s[2:3]
	v_or_b32_e32 v3, 64, v6
	v_cmp_gt_i32_e64 s[2:3], s9, v3
	s_and_b64 s[6:7], s[14:15], s[2:3]
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_add_u32_e32 v8, v3, v254
	v_ashrrev_i32_e32 v9, 31, v8
	v_cvt_pk_f16_f32 v4, v246, v247
	v_cvt_pk_f16_f32 v5, v248, v249
	v_lshl_add_u64 v[8:9], v[8:9], 1, v[0:1]
	global_store_dwordx2 v[8:9], v[4:5], off
.LBB0_42:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v4, 0x60, v6
	v_cmp_gt_i32_e64 s[4:5], s9, v4
	s_and_b64 s[10:11], s[14:15], s[4:5]
	s_and_saveexec_b64 s[6:7], s[10:11]
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_add_u32_e32 v10, v4, v254
	v_ashrrev_i32_e32 v11, 31, v10
	v_cvt_pk_f16_f32 v8, v242, v243
	v_cvt_pk_f16_f32 v9, v244, v245
	v_lshl_add_u64 v[10:11], v[10:11], 1, v[0:1]
	global_store_dwordx2 v[10:11], v[8:9], off
.LBB0_44:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v5, 0x80, v6
	v_cmp_gt_i32_e64 s[6:7], s9, v5
	s_and_b64 s[16:17], s[14:15], s[6:7]
	s_and_saveexec_b64 s[10:11], s[16:17]
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_add_u32_e32 v10, v5, v254
	v_ashrrev_i32_e32 v11, 31, v10
	v_cvt_pk_f16_f32 v8, v238, v239
	v_cvt_pk_f16_f32 v9, v240, v241
	v_lshl_add_u64 v[10:11], v[10:11], 1, v[0:1]
	global_store_dwordx2 v[10:11], v[8:9], off
.LBB0_46:
	s_or_b64 exec, exec, s[10:11]
	v_or_b32_e32 v8, 0xa0, v6
	v_cmp_gt_i32_e64 s[18:19], s9, v8
	s_and_b64 s[16:17], s[14:15], s[18:19]
	s_and_saveexec_b64 s[10:11], s[16:17]
	s_cbranch_execz .LBB0_48
; %bb.47:
	v_add_u32_e32 v12, v8, v254
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v234, v235
	v_cvt_pk_f16_f32 v11, v236, v237
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
.LBB0_48:
	s_or_b64 exec, exec, s[10:11]
	v_or_b32_e32 v9, 0xc0, v6
	v_cmp_gt_i32_e64 s[10:11], s9, v9
	s_and_b64 s[20:21], s[14:15], s[10:11]
	s_and_saveexec_b64 s[16:17], s[20:21]
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_add_u32_e32 v12, v9, v254
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v230, v231
	v_cvt_pk_f16_f32 v11, v232, v233
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
.LBB0_50:
	s_or_b64 exec, exec, s[16:17]
	v_or_b32_e32 v230, 0xe0, v6
	v_cmp_gt_i32_e64 s[16:17], s9, v230
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_add_u32_e32 v12, v230, v254
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v226, v227
	v_cvt_pk_f16_f32 v11, v228, v229
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
.LBB0_52:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v10, 32, v7
	v_cmp_gt_i32_e64 s[14:15], s8, v10
	v_mul_lo_u32 v226, v10, s13
	s_and_b64 s[22:23], s[14:15], vcc
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_115
; %bb.53:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_116
.LBB0_54:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_117
.LBB0_55:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_118
.LBB0_56:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_119
.LBB0_57:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_120
.LBB0_58:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_121
.LBB0_59:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_61
.LBB0_60:
	v_add_u32_e32 v12, v226, v230
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v194, v195
	v_cvt_pk_f16_f32 v11, v196, v197
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
.LBB0_61:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v10, 64, v7
	v_cmp_gt_i32_e64 s[14:15], s8, v10
	v_mul_lo_u32 v194, v10, s13
	s_and_b64 s[22:23], s[14:15], vcc
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_122
; %bb.62:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_123
.LBB0_63:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_124
.LBB0_64:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_125
.LBB0_65:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_126
.LBB0_66:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_127
.LBB0_67:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_128
.LBB0_68:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_70
.LBB0_69:
	v_add_u32_e32 v12, v194, v230
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v162, v163
	v_cvt_pk_f16_f32 v11, v164, v165
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
.LBB0_70:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v10, 0x60, v7
	v_cmp_gt_i32_e64 s[14:15], s8, v10
	v_mul_lo_u32 v162, v10, s13
	s_and_b64 s[22:23], s[14:15], vcc
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_129
; %bb.71:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_130
.LBB0_72:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_131
.LBB0_73:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_132
.LBB0_74:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_133
.LBB0_75:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_134
.LBB0_76:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_135
.LBB0_77:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_79
.LBB0_78:
	v_add_u32_e32 v12, v162, v230
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v130, v131
	v_cvt_pk_f16_f32 v11, v132, v133
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
.LBB0_79:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v10, 0x80, v7
	v_cmp_gt_i32_e64 s[14:15], s8, v10
	v_mul_lo_u32 v130, v10, s13
	s_and_b64 s[22:23], s[14:15], vcc
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_136
; %bb.80:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_137
.LBB0_81:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_138
.LBB0_82:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_139
.LBB0_83:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_140
.LBB0_84:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_141
.LBB0_85:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_142
.LBB0_86:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_88
.LBB0_87:
	v_add_u32_e32 v12, v130, v230
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v98, v99
	v_cvt_pk_f16_f32 v11, v100, v101
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
.LBB0_88:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v10, 0xa0, v7
	v_cmp_gt_i32_e64 s[14:15], s8, v10
	v_mul_lo_u32 v98, v10, s13
	s_and_b64 s[22:23], s[14:15], vcc
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_143
; %bb.89:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_144
.LBB0_90:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_145
.LBB0_91:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_146
.LBB0_92:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_147
.LBB0_93:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_148
.LBB0_94:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_149
.LBB0_95:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_97
.LBB0_96:
	v_add_u32_e32 v12, v98, v230
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v66, v67
	v_cvt_pk_f16_f32 v11, v68, v69
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
.LBB0_97:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v10, 0xc0, v7
	v_cmp_gt_i32_e64 s[14:15], s8, v10
	v_mul_lo_u32 v66, v10, s13
	s_and_b64 s[22:23], s[14:15], vcc
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_150
; %bb.98:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_151
.LBB0_99:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_152
.LBB0_100:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_153
.LBB0_101:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_154
.LBB0_102:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_155
.LBB0_103:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_156
.LBB0_104:
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_106
.LBB0_105:
	v_add_u32_e32 v12, v66, v230
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v34, v35
	v_cvt_pk_f16_f32 v11, v36, v37
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
.LBB0_106:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v10, 0xe0, v7
	v_cmp_gt_i32_e64 s[14:15], s8, v10
	v_mul_lo_u32 v7, v10, s13
	s_and_b64 s[12:13], s[14:15], vcc
	s_and_saveexec_b64 s[8:9], s[12:13]
	s_cbranch_execnz .LBB0_157
; %bb.107:
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[8:9], s[14:15], s[0:1]
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execnz .LBB0_158
.LBB0_108:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[2:3]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_159
.LBB0_109:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[4:5]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_160
.LBB0_110:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[6:7]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_161
.LBB0_111:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[18:19]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_162
.LBB0_112:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[10:11]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_163
.LBB0_113:
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[0:1], s[14:15], s[16:17]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execnz .LBB0_164
.LBB0_114:
	s_endpgm
.LBB0_115:
	v_add_u32_e32 v12, v226, v6
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v222, v223
	v_cvt_pk_f16_f32 v11, v224, v225
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_54
.LBB0_116:
	v_add_u32_e32 v12, v226, v2
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v218, v219
	v_cvt_pk_f16_f32 v11, v220, v221
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_55
.LBB0_117:
	v_add_u32_e32 v12, v226, v3
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v214, v215
	v_cvt_pk_f16_f32 v11, v216, v217
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_56
.LBB0_118:
	v_add_u32_e32 v12, v226, v4
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v210, v211
	v_cvt_pk_f16_f32 v11, v212, v213
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_57
.LBB0_119:
	v_add_u32_e32 v12, v226, v5
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v206, v207
	v_cvt_pk_f16_f32 v11, v208, v209
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_58
.LBB0_120:
	v_add_u32_e32 v12, v226, v8
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v202, v203
	v_cvt_pk_f16_f32 v11, v204, v205
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_59
.LBB0_121:
	v_add_u32_e32 v12, v226, v9
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v198, v199
	v_cvt_pk_f16_f32 v11, v200, v201
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_60
	s_branch .LBB0_61
.LBB0_122:
	v_add_u32_e32 v12, v194, v6
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v190, v191
	v_cvt_pk_f16_f32 v11, v192, v193
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_63
.LBB0_123:
	v_add_u32_e32 v12, v194, v2
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v186, v187
	v_cvt_pk_f16_f32 v11, v188, v189
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_64
.LBB0_124:
	v_add_u32_e32 v12, v194, v3
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v182, v183
	v_cvt_pk_f16_f32 v11, v184, v185
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_65
.LBB0_125:
	v_add_u32_e32 v12, v194, v4
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v178, v179
	v_cvt_pk_f16_f32 v11, v180, v181
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_66
.LBB0_126:
	v_add_u32_e32 v12, v194, v5
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v174, v175
	v_cvt_pk_f16_f32 v11, v176, v177
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_67
.LBB0_127:
	v_add_u32_e32 v12, v194, v8
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v170, v171
	v_cvt_pk_f16_f32 v11, v172, v173
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_68
.LBB0_128:
	v_add_u32_e32 v12, v194, v9
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v166, v167
	v_cvt_pk_f16_f32 v11, v168, v169
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_69
	s_branch .LBB0_70
.LBB0_129:
	v_add_u32_e32 v12, v162, v6
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v158, v159
	v_cvt_pk_f16_f32 v11, v160, v161
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_72
.LBB0_130:
	v_add_u32_e32 v12, v162, v2
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v154, v155
	v_cvt_pk_f16_f32 v11, v156, v157
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_73
.LBB0_131:
	v_add_u32_e32 v12, v162, v3
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v150, v151
	v_cvt_pk_f16_f32 v11, v152, v153
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_74
.LBB0_132:
	v_add_u32_e32 v12, v162, v4
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v146, v147
	v_cvt_pk_f16_f32 v11, v148, v149
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_75
.LBB0_133:
	v_add_u32_e32 v12, v162, v5
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v142, v143
	v_cvt_pk_f16_f32 v11, v144, v145
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_76
.LBB0_134:
	v_add_u32_e32 v12, v162, v8
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v138, v139
	v_cvt_pk_f16_f32 v11, v140, v141
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_77
.LBB0_135:
	v_add_u32_e32 v12, v162, v9
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v134, v135
	v_cvt_pk_f16_f32 v11, v136, v137
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_78
	s_branch .LBB0_79
.LBB0_136:
	v_add_u32_e32 v12, v130, v6
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v126, v127
	v_cvt_pk_f16_f32 v11, v128, v129
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_81
.LBB0_137:
	v_add_u32_e32 v12, v130, v2
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v122, v123
	v_cvt_pk_f16_f32 v11, v124, v125
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_82
.LBB0_138:
	v_add_u32_e32 v12, v130, v3
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v118, v119
	v_cvt_pk_f16_f32 v11, v120, v121
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_83
.LBB0_139:
	v_add_u32_e32 v12, v130, v4
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v114, v115
	v_cvt_pk_f16_f32 v11, v116, v117
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_84
.LBB0_140:
	v_add_u32_e32 v12, v130, v5
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v110, v111
	v_cvt_pk_f16_f32 v11, v112, v113
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_85
.LBB0_141:
	v_add_u32_e32 v12, v130, v8
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v106, v107
	v_cvt_pk_f16_f32 v11, v108, v109
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_86
.LBB0_142:
	v_add_u32_e32 v12, v130, v9
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v102, v103
	v_cvt_pk_f16_f32 v11, v104, v105
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_87
	s_branch .LBB0_88
.LBB0_143:
	v_add_u32_e32 v12, v98, v6
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v94, v95
	v_cvt_pk_f16_f32 v11, v96, v97
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_90
.LBB0_144:
	v_add_u32_e32 v12, v98, v2
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v90, v91
	v_cvt_pk_f16_f32 v11, v92, v93
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_91
.LBB0_145:
	v_add_u32_e32 v12, v98, v3
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v86, v87
	v_cvt_pk_f16_f32 v11, v88, v89
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_92
.LBB0_146:
	v_add_u32_e32 v12, v98, v4
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v82, v83
	v_cvt_pk_f16_f32 v11, v84, v85
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_93
.LBB0_147:
	v_add_u32_e32 v12, v98, v5
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v78, v79
	v_cvt_pk_f16_f32 v11, v80, v81
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_94
.LBB0_148:
	v_add_u32_e32 v12, v98, v8
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v74, v75
	v_cvt_pk_f16_f32 v11, v76, v77
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_95
.LBB0_149:
	v_add_u32_e32 v12, v98, v9
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v70, v71
	v_cvt_pk_f16_f32 v11, v72, v73
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_96
	s_branch .LBB0_97
.LBB0_150:
	v_add_u32_e32 v12, v66, v6
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v58, v59
	v_cvt_pk_f16_f32 v11, v60, v61
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[0:1]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_99
.LBB0_151:
	v_add_u32_e32 v12, v66, v2
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v50, v51
	v_cvt_pk_f16_f32 v11, v52, v53
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[2:3]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_100
.LBB0_152:
	v_add_u32_e32 v12, v66, v3
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v42, v43
	v_cvt_pk_f16_f32 v11, v44, v45
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[4:5]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_101
.LBB0_153:
	v_add_u32_e32 v12, v66, v4
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v62, v63
	v_cvt_pk_f16_f32 v11, v64, v65
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[6:7]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_102
.LBB0_154:
	v_add_u32_e32 v12, v66, v5
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v54, v55
	v_cvt_pk_f16_f32 v11, v56, v57
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_103
.LBB0_155:
	v_add_u32_e32 v12, v66, v8
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v46, v47
	v_cvt_pk_f16_f32 v11, v48, v49
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[22:23], s[14:15], s[10:11]
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_104
.LBB0_156:
	v_add_u32_e32 v12, v66, v9
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v38, v39
	v_cvt_pk_f16_f32 v11, v40, v41
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[20:21]
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_105
	s_branch .LBB0_106
.LBB0_157:
	v_add_u32_e32 v12, v7, v6
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v30, v31
	v_cvt_pk_f16_f32 v11, v32, v33
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[8:9], s[14:15], s[0:1]
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB0_108
.LBB0_158:
	v_add_u32_e32 v12, v7, v2
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_pk_f16_f32 v10, v26, v27
	v_cvt_pk_f16_f32 v11, v28, v29
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[0:1]
	global_store_dwordx2 v[12:13], v[10:11], off
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[2:3]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_109
.LBB0_159:
	v_add_u32_e32 v2, v7, v3
	v_ashrrev_i32_e32 v3, 31, v2
	v_cvt_pk_f16_f32 v10, v22, v23
	v_cvt_pk_f16_f32 v11, v24, v25
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_dwordx2 v[2:3], v[10:11], off
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[4:5]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_110
.LBB0_160:
	v_add_u32_e32 v10, v7, v4
	v_ashrrev_i32_e32 v11, 31, v10
	v_cvt_pk_f16_f32 v2, v18, v19
	v_cvt_pk_f16_f32 v3, v20, v21
	v_lshl_add_u64 v[10:11], v[10:11], 1, v[0:1]
	global_store_dwordx2 v[10:11], v[2:3], off
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[6:7]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_111
.LBB0_161:
	v_add_u32_e32 v4, v7, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_cvt_pk_f16_f32 v2, v14, v15
	v_cvt_pk_f16_f32 v3, v16, v17
	v_lshl_add_u64 v[4:5], v[4:5], 1, v[0:1]
	global_store_dwordx2 v[4:5], v[2:3], off
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[18:19]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_112
.LBB0_162:
	v_accvgpr_read_b32 v2, a132
	v_accvgpr_read_b32 v3, a133
	v_accvgpr_read_b32 v4, a134
	v_accvgpr_read_b32 v5, a135
	v_cvt_pk_f16_f32 v2, v2, v3
	v_cvt_pk_f16_f32 v3, v4, v5
	v_add_u32_e32 v4, v7, v8
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, v[0:1]
	global_store_dwordx2 v[4:5], v[2:3], off
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[2:3], s[14:15], s[10:11]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_113
.LBB0_163:
	v_accvgpr_read_b32 v2, a128
	v_accvgpr_read_b32 v3, a129
	v_accvgpr_read_b32 v4, a130
	v_accvgpr_read_b32 v5, a131
	v_cvt_pk_f16_f32 v2, v2, v3
	v_cvt_pk_f16_f32 v3, v4, v5
	v_add_u32_e32 v4, v7, v9
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, v[0:1]
	global_store_dwordx2 v[4:5], v[2:3], off
	s_or_b64 exec, exec, s[0:1]
	s_and_b64 s[0:1], s[14:15], s[16:17]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_114
.LBB0_164:
	v_accvgpr_read_b32 v2, a124
	v_accvgpr_read_b32 v3, a125
	v_accvgpr_read_b32 v4, a126
	v_accvgpr_read_b32 v5, a127
	v_cvt_pk_f16_f32 v2, v2, v3
	v_cvt_pk_f16_f32 v3, v4, v5
	v_add_u32_e32 v4, v7, v230
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[0:1], v[4:5], 1, v[0:1]
	global_store_dwordx2 v[0:1], v[2:3], off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v0_naive
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
		.amdhsa_next_free_vgpr 430
		.amdhsa_next_free_sgpr 24
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
	.size	v0_naive, .Lfunc_end0-v0_naive
	.cfi_endproc
                                        ; -- End function
	.set v0_naive.num_vgpr, 256
	.set v0_naive.num_agpr, 174
	.set v0_naive.numbered_sgpr, 24
	.set v0_naive.num_named_barrier, 0
	.set v0_naive.private_seg_size, 0
	.set v0_naive.uses_vcc, 1
	.set v0_naive.uses_flat_scratch, 0
	.set v0_naive.has_dyn_sized_stack, 0
	.set v0_naive.has_recursion, 0
	.set v0_naive.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11336
; TotalNumSgprs: 30
; NumVgprs: 256
; NumAgprs: 174
; TotalNumVgprs: 430
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 53
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 430
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
	.byte	15                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	59                              ; DW_AT_call_line
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
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7
.Linfo_string2:
	.asciz	"/root/gfx9-gluon-tutorials/kernels/gemm/a16w16/v0_naive" ; string offset=24
.Linfo_string3:
	.asciz	"v0_naive"                      ; string offset=80
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     174
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
    .name:           v0_naive
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         v0_naive.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     430
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
