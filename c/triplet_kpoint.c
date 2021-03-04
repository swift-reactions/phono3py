/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* These codes were originally parts of spglib, but only develped */
/* and used for phono3py. Therefore these were moved from spglib to */
/* phono3py. This file is part of phonopy. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the phonopy project nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
/* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
/* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS */
/* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE */
/* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, */
/* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, */
/* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; */
/* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER */
/* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT */
/* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
/* POSSIBILITY OF SUCH DAMAGE. */

#include <stddef.h>
#include <stdlib.h>
#include "kpoint.h"
#include "grgrid.h"
#include "triplet.h"
#include "triplet_kpoint.h"

#define KPT_NUM_BZ_SEARCH_SPACE 125
static long bz_search_space[KPT_NUM_BZ_SEARCH_SPACE][3] = {
  { 0,  0,  0},
  { 0,  0,  1},
  { 0,  0,  2},
  { 0,  0, -2},
  { 0,  0, -1},
  { 0,  1,  0},
  { 0,  1,  1},
  { 0,  1,  2},
  { 0,  1, -2},
  { 0,  1, -1},
  { 0,  2,  0},
  { 0,  2,  1},
  { 0,  2,  2},
  { 0,  2, -2},
  { 0,  2, -1},
  { 0, -2,  0},
  { 0, -2,  1},
  { 0, -2,  2},
  { 0, -2, -2},
  { 0, -2, -1},
  { 0, -1,  0},
  { 0, -1,  1},
  { 0, -1,  2},
  { 0, -1, -2},
  { 0, -1, -1},
  { 1,  0,  0},
  { 1,  0,  1},
  { 1,  0,  2},
  { 1,  0, -2},
  { 1,  0, -1},
  { 1,  1,  0},
  { 1,  1,  1},
  { 1,  1,  2},
  { 1,  1, -2},
  { 1,  1, -1},
  { 1,  2,  0},
  { 1,  2,  1},
  { 1,  2,  2},
  { 1,  2, -2},
  { 1,  2, -1},
  { 1, -2,  0},
  { 1, -2,  1},
  { 1, -2,  2},
  { 1, -2, -2},
  { 1, -2, -1},
  { 1, -1,  0},
  { 1, -1,  1},
  { 1, -1,  2},
  { 1, -1, -2},
  { 1, -1, -1},
  { 2,  0,  0},
  { 2,  0,  1},
  { 2,  0,  2},
  { 2,  0, -2},
  { 2,  0, -1},
  { 2,  1,  0},
  { 2,  1,  1},
  { 2,  1,  2},
  { 2,  1, -2},
  { 2,  1, -1},
  { 2,  2,  0},
  { 2,  2,  1},
  { 2,  2,  2},
  { 2,  2, -2},
  { 2,  2, -1},
  { 2, -2,  0},
  { 2, -2,  1},
  { 2, -2,  2},
  { 2, -2, -2},
  { 2, -2, -1},
  { 2, -1,  0},
  { 2, -1,  1},
  { 2, -1,  2},
  { 2, -1, -2},
  { 2, -1, -1},
  {-2,  0,  0},
  {-2,  0,  1},
  {-2,  0,  2},
  {-2,  0, -2},
  {-2,  0, -1},
  {-2,  1,  0},
  {-2,  1,  1},
  {-2,  1,  2},
  {-2,  1, -2},
  {-2,  1, -1},
  {-2,  2,  0},
  {-2,  2,  1},
  {-2,  2,  2},
  {-2,  2, -2},
  {-2,  2, -1},
  {-2, -2,  0},
  {-2, -2,  1},
  {-2, -2,  2},
  {-2, -2, -2},
  {-2, -2, -1},
  {-2, -1,  0},
  {-2, -1,  1},
  {-2, -1,  2},
  {-2, -1, -2},
  {-2, -1, -1},
  {-1,  0,  0},
  {-1,  0,  1},
  {-1,  0,  2},
  {-1,  0, -2},
  {-1,  0, -1},
  {-1,  1,  0},
  {-1,  1,  1},
  {-1,  1,  2},
  {-1,  1, -2},
  {-1,  1, -1},
  {-1,  2,  0},
  {-1,  2,  1},
  {-1,  2,  2},
  {-1,  2, -2},
  {-1,  2, -1},
  {-1, -2,  0},
  {-1, -2,  1},
  {-1, -2,  2},
  {-1, -2, -2},
  {-1, -2, -1},
  {-1, -1,  0},
  {-1, -1,  1},
  {-1, -1,  2},
  {-1, -1, -2},
  {-1, -1, -1}
};

static void grid_point_to_address_double(long address_double[3],
                                         const long grid_point,
                                         const long mesh[3],
                                         const long is_shift[3]);
static long get_ir_triplets_at_q(long *map_triplets,
                                 long *map_q,
                                 long (*grid_address)[3],
                                 const long grid_point,
                                 const long mesh[3],
                                 const MatLONG * rot_reciprocal,
                                 const long swappable);
static long get_BZ_triplets_at_q(long (*triplets)[3],
                                 const long grid_point,
                                 TPLCONST long (*bz_grid_address)[3],
                                 const long *bz_map,
                                 const long *map_triplets,
                                 const long num_map_triplets,
                                 const long mesh[3]);
static long get_third_q_of_triplets_at_q(long bz_address[3][3],
                                         const long q_index,
                                         const long *bz_map,
                                         const long mesh[3],
                                         const long bzmesh[3]);
static void modulo_l3(long v[3], const long m[3]);

long tpk_get_ir_triplets_at_q(long *map_triplets,
                              long *map_q,
                              long (*grid_address)[3],
                              const long grid_point,
                              const long mesh[3],
                              const long is_time_reversal,
                              const MatLONG * rotations,
                              const long swappable)
{
  long num_ir;
  MatLONG *rot_reciprocal;

  rot_reciprocal = kpt_get_point_group_reciprocal(rotations, is_time_reversal);
  num_ir = get_ir_triplets_at_q(map_triplets,
                                map_q,
                                grid_address,
                                grid_point,
                                mesh,
                                rot_reciprocal,
                                swappable);
  kpt_free_MatLONG(rot_reciprocal);
  return num_ir;
}

long tpk_get_BZ_triplets_at_q(long (*triplets)[3],
                              const long grid_point,
                              TPLCONST long (*bz_grid_address)[3],
                              const long *bz_map,
                              const long *map_triplets,
                              const long num_map_triplets,
                              const long mesh[3])
{
  return get_BZ_triplets_at_q(triplets,
                              grid_point,
                              bz_grid_address,
                              bz_map,
                              map_triplets,
                              num_map_triplets,
                              mesh);
}

static long get_ir_triplets_at_q(long *map_triplets,
                                 long *map_q,
                                 long (*grid_address)[3],
                                 const long grid_point,
                                 const long mesh[3],
                                 const MatLONG * rot_reciprocal,
                                 const long swappable)
{
  long i, j, num_grid, q_2, num_ir_q, num_ir_triplets, ir_gp;
  long mesh_double[3], is_shift[3];
  long address_double0[3], address_double1[3], address_double2[3];
  long *ir_grid_points, *third_q;
  double tolerance;
  double stabilizer_q[1][3];
  MatLONG *rot_reciprocal_q;

  ir_grid_points = NULL;
  third_q = NULL;
  rot_reciprocal_q = NULL;
  num_ir_triplets = 0;

  tolerance = 0.01 / (mesh[0] + mesh[1] + mesh[2]);
  num_grid = mesh[0] * mesh[1] * (long)mesh[2];

  for (i = 0; i < 3; i++) {
    /* Only consider the gamma-point */
    is_shift[i] = 0;
    mesh_double[i] = mesh[i] * 2;
  }

  /* Search irreducible q-points (map_q) with a stabilizer */
  /* q */
  grid_point_to_address_double(address_double0, grid_point, mesh, is_shift);
  for (i = 0; i < 3; i++) {
    stabilizer_q[0][i] =
      (double)address_double0[i] / mesh_double[i] - (address_double0[i] > mesh[i]);
  }

  rot_reciprocal_q = kpt_get_point_group_reciprocal_with_q(rot_reciprocal,
                                                           tolerance,
                                                           1,
                                                           stabilizer_q);
  num_ir_q = kpt_get_irreducible_reciprocal_mesh(grid_address,
                                                 map_q,
                                                 mesh,
                                                 is_shift,
                                                 rot_reciprocal_q);
  kpt_free_MatLONG(rot_reciprocal_q);
  rot_reciprocal_q = NULL;

  if ((third_q = (long*) malloc(sizeof(long) * num_ir_q)) == NULL) {
    warning_print("Memory could not be allocated.");
    goto ret;
  }

  if ((ir_grid_points = (long*) malloc(sizeof(long) * num_ir_q)) == NULL) {
    warning_print("Memory could not be allocated.");
    goto ret;
  }

  num_ir_q = 0;
  for (i = 0; i < num_grid; i++) {
    if (map_q[i] == i) {
      ir_grid_points[num_ir_q] = i;
      num_ir_q++;
    }
  }

  for (i = 0; i < num_grid; i++) {
    map_triplets[i] = num_grid;  /* When not found, map_triplets == num_grid */
  }

#pragma omp parallel for private(j, address_double1, address_double2)
  for (i = 0; i < num_ir_q; i++) {
    grid_point_to_address_double(address_double1,
                                 ir_grid_points[i],
                                 mesh,
                                 is_shift); /* q' */
    for (j = 0; j < 3; j++) { /* q'' */
      address_double2[j] = - address_double0[j] - address_double1[j];
    }
    third_q[i] = grg_get_double_grid_index(address_double2, mesh, is_shift);
  }

  if (swappable) { /* search q1 <-> q2 */
    for (i = 0; i < num_ir_q; i++) {
      ir_gp = ir_grid_points[i];
      q_2 = third_q[i];
      if (map_triplets[map_q[q_2]] < num_grid) {
        map_triplets[ir_gp] = map_triplets[map_q[q_2]];
      } else {
        map_triplets[ir_gp] = ir_gp;
        num_ir_triplets++;
      }
    }
  } else {
    for (i = 0; i < num_ir_q; i++) {
      ir_gp = ir_grid_points[i];
      map_triplets[ir_gp] = ir_gp;
      num_ir_triplets++;
    }
  }

#pragma omp parallel for
  for (i = 0; i < num_grid; i++) {
    map_triplets[i] = map_triplets[map_q[i]];
  }

ret:
  if (third_q) {
    free(third_q);
    third_q = NULL;
  }
  if (ir_grid_points) {
    free(ir_grid_points);
    ir_grid_points = NULL;
  }
  return num_ir_triplets;
}

static long get_BZ_triplets_at_q(long (*triplets)[3],
                                 const long grid_point,
                                 TPLCONST long (*bz_grid_address)[3],
                                 const long *bz_map,
                                 const long *map_triplets,
                                 const long num_map_triplets,
                                 const long mesh[3])
{
  long i, num_ir;
  long j, k;
  long bz_address[3][3], bz_address_double[3], bzmesh[3], PS[3];
  long *ir_grid_points;

  ir_grid_points = NULL;

  for (i = 0; i < 3; i++) {
    PS[i] = 0;
    bzmesh[i] = mesh[i] * 2;
  }

  num_ir = 0;

  if ((ir_grid_points = (long*) malloc(sizeof(long) * num_map_triplets))
      == NULL) {
    warning_print("Memory could not be allocated.");
    goto ret;
  }

  for (i = 0; i < num_map_triplets; i++) {
    if (map_triplets[i] == i) {
      ir_grid_points[num_ir] = i;
      num_ir++;
    }
  }

#pragma omp parallel for private(j, k, bz_address, bz_address_double)
  for (i = 0; i < num_ir; i++) {
    for (j = 0; j < 3; j++) {
      bz_address[0][j] = bz_grid_address[grid_point][j];
      bz_address[1][j] = bz_grid_address[ir_grid_points[i]][j];
      bz_address[2][j] = - bz_address[0][j] - bz_address[1][j];
    }
    for (j = 2; j > -1; j--) {
      if (get_third_q_of_triplets_at_q(bz_address,
                                       j,
                                       bz_map,
                                       mesh,
                                       bzmesh) == 0) {
        break;
      }
    }
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
        bz_address_double[k] = bz_address[j][k] * 2;
      }
      triplets[i][j] =
        bz_map[grg_get_double_grid_index(bz_address_double, bzmesh, PS)];
    }
  }

  free(ir_grid_points);
  ir_grid_points = NULL;

ret:
  return num_ir;
}

static long get_third_q_of_triplets_at_q(long bz_address[3][3],
                                         const long q_index,
                                         const long *bz_map,
                                         const long mesh[3],
                                         const long bzmesh[3])
{
  long i, j, smallest_g, smallest_index, sum_g, delta_g[3];
  long prod_bzmesh;
  long bzgp[KPT_NUM_BZ_SEARCH_SPACE];
  long bz_address_double[3], PS[3];

  prod_bzmesh = (long)bzmesh[0] * bzmesh[1] * bzmesh[2];

  modulo_l3(bz_address[q_index], mesh);
  for (i = 0; i < 3; i++) {
    PS[i] = 0;
    delta_g[i] = 0;
    for (j = 0; j < 3; j++) {
      delta_g[i] += bz_address[j][i];
    }
    delta_g[i] /= mesh[i];
  }

  for (i = 0; i < KPT_NUM_BZ_SEARCH_SPACE; i++) {
    for (j = 0; j < 3; j++) {
      bz_address_double[j] = (bz_address[q_index][j] +
                              bz_search_space[i][j] * mesh[j]) * 2;
    }
    bzgp[i] = bz_map[grg_get_double_grid_index(bz_address_double, bzmesh, PS)];
  }

  for (i = 0; i < KPT_NUM_BZ_SEARCH_SPACE; i++) {
    if (bzgp[i] != prod_bzmesh) {
      goto escape;
    }
  }

escape:

  smallest_g = 4;
  smallest_index = 0;

  for (i = 0; i < KPT_NUM_BZ_SEARCH_SPACE; i++) {
    if (bzgp[i] < prod_bzmesh) { /* q'' is in BZ */
      sum_g = (labs(delta_g[0] + bz_search_space[i][0]) +
               labs(delta_g[1] + bz_search_space[i][1]) +
               labs(delta_g[2] + bz_search_space[i][2]));
      if (sum_g < smallest_g) {
        smallest_index = i;
        smallest_g = sum_g;
      }
    }
  }

  for (i = 0; i < 3; i++) {
    bz_address[q_index][i] += bz_search_space[smallest_index][i] * mesh[i];
  }

  return smallest_g;
}

static void grid_point_to_address_double(long address_double[3],
                                         const long grid_point,
                                         const long mesh[3],
                                         const long is_shift[3])
{
  long i;
  long address[3];

#ifndef GRID_ORDER_XYZ
  address[2] = grid_point / (mesh[0] * mesh[1]);
  address[1] = (grid_point - address[2] * mesh[0] * mesh[1]) / mesh[0];
  address[0] = grid_point % mesh[0];
#else
  address[0] = grid_point / (mesh[1] * mesh[2]);
  address[1] = (grid_point - address[0] * mesh[1] * mesh[2]) / mesh[2];
  address[2] = grid_point % mesh[2];
#endif

  for (i = 0; i < 3; i++) {
    address_double[i] = address[i] * 2 + is_shift[i];
  }
}

static void modulo_l3(long v[3], const long m[3])
{
  long i;

  for (i = 0; i < 3; i++) {
    v[i] = v[i] % m[i];

    if (v[i] < 0) {
      v[i] += m[i];
    }
  }
}
