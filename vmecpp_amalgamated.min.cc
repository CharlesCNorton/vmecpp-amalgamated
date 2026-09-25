// ============================================================================
// VMEC++ - single-file C++ amalgamation, presentation-stripped layer
//
// The same program as vmecpp_amalgamated.cc, with comments and indentation
// removed and the per-file SPDX and copyright headers replaced by the single
// notice below. Read this layer to take in the whole of VMEC++ at once; the
// source: and header: markers are kept, so any region maps back to that file,
// where the comments on its routines are.
//
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// VMEC++ (github.com/proximafusion/vmecpp): MIT License, Copyright (c)
// 2024-present Proxima Fusion GmbH. abscab
// (github.com/jonathanschilling/abscab-cpp), the Biot-Savart routines the
// free-boundary path uses: Apache License 2.0, Copyright (c) Jonathan
// Schilling. See LICENSE, NOTICE and THIRD_PARTY_LICENSES/.
//
// Unofficial redistribution; not affiliated with or endorsed by Proxima Fusion.
//
// Provenance: github.com/proximafusion/vmecpp v0.7.5-20-gd66698e9
//
// Scope matches vmecpp_amalgamated.cc exactly: the whole solver, fixed and
// free boundary, every profile parameterization, the complete output suite and
// the standalone main(), less the FFTX/SPIRAL transform (VMECPP_USE_FFTX) and
// the Enzyme autodiff translation units (VMECPP_ENABLE_ENZYME).
//
// Build:
//   cmake --build build --target vmecpp_min
//   ./build/vmecpp_min input.json [n_threads]   # writes input.out.h5
// ============================================================================
// source: abscab/abscab.cc

// header: abscab/abscab.hh

#ifndef ABSCAB_H
#define ABSCAB_H

#ifdef _OPENMP
#include <omp.h>
#endif

namespace abscab {

const double MU_0 = 1.25663706212e-6;

double cel(double k_c, double p, double a, double b);

double sws_A_z_ax_f(double zP);

double sws_A_z_ax_n(double zP);

double sws_A_z_ax(double zP);

double sws_A_z_rad_f(double rhoP);

double sws_A_z_rad_n(double rhoP);

double sws_A_z_rad(double rhoP);

double sws_A_z_f(double rhoP, double zP);

double sws_A_z_n(double rhoP, double zP);

double sws_B_phi_rad(double rhoP);

double sws_B_phi_f(double rhoP, double zP);

double sws_B_phi_n(double rhoP, double zP);

double cwl_A_phi_f(double rhoP, double zP);

double cwl_A_phi_n(double rhoP, double zP);

double cwl_A_phi_v(double zP);

double cwl_B_rho_f(double rhoP, double zP);

double cwl_B_rho_n(double rhoP, double zP);

double cwl_B_rho_v(double zP);

double cwl_B_z_f1(double rhoP, double zP);

double cwl_B_z_f2(double rhoP, double zP);

double cwl_B_z_n(double rhoP, double zP);

double cwl_B_z_v(double zP);

double straightWireSegment_A_z(double rhoP, double zP);

double straightWireSegment_B_phi(double rhoP, double zP);

double circularWireLoop_A_phi(double rhoP, double zP);

double circularWireLoop_B_rho(double rhoP, double zP);

double circularWireLoop_B_z(double rhoP, double zP);

void vectorPotentialCircularFilament(double *center, double *normal, double radius,
double current, int nEvalPos, double *evalPos, double *vectorPotential);

void magneticFieldCircularFilament(double *center, double *normal, double radius,
double current, int nEvalPos, double *evalPos, double *magneticField);

void kernelVectorPotentialPolygonFilament(
double *vertices,
double current,
double *evalPos,
double *vectorPotential,
int idxSourceStart,
int idxSourceEnd,
int idxEvalStart,
int idxEvalEnd,
bool useCompensatedSummation);

void kernelVectorPotentialPolygonFilament(
void (*vertexSupplier)(int i, double *point),
double current,
double *evalPos,
double *vectorPotential,
int idxSourceStart,
int idxSourceEnd,
int idxEvalStart,
int idxEvalEnd,
bool useCompensatedSummation);

void kernelMagneticFieldPolygonFilament(
double *vertices,
double current,
double *evalPos,
double *magneticField,
int idxSourceStart,
int idxSourceEnd,
int idxEvalStart,
int idxEvalEnd,
bool useCompensatedSummation);

void kernelMagneticFieldPolygonFilament(
void (*vertexSupplier)(int i, double *point),
double current,
double *evalPos,
double *magneticField,
int idxSourceStart,
int idxSourceEnd,
int idxEvalStart,
int idxEvalEnd,
bool useCompensatedSummation);

void vectorPotentialPolygonFilament(
int numVertices,
double *vertices,
double current,
int numEvalPos,
double *evalPos,
double *vectorPotential,
#ifdef _OPENMP
int numProcessors = omp_get_max_threads(),
#else
int numProcessors = 1,
#endif
bool useCompensatedSummation = true);

void vectorPotentialPolygonFilament(
int numVertices,
void (*vertexSupplier)(int i, double *point),
double current,
int numEvalPos,
double *evalPos,
double *vectorPotential,
#ifdef _OPENMP
int numProcessors = omp_get_max_threads(),
#else
int numProcessors = 1,
#endif
bool useCompensatedSummation = true);

void magneticFieldPolygonFilament(
int numVertices,
double *vertices,
double current,
int numEvalPos,
double *evalPos,
double *magneticField,
#ifdef _OPENMP
int numProcessors = omp_get_max_threads(),
#else
int numProcessors = 1,
#endif
bool useCompensatedSummation = true);

void magneticFieldPolygonFilament(
int numVertices,
void (*vertexSupplier)(int i, double *point),
double current,
int numEvalPos,
double *evalPos,
double *magneticField,
#ifdef _OPENMP
int numProcessors = omp_get_max_threads(),
#else
int numProcessors = 1,
#endif
bool useCompensatedSummation = true);

}

#endif

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

namespace abscab {

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

const double MU_0_BY_PI = MU_0 / M_PI;

const double MU_0_BY_2_PI = MU_0 / (2.0 * M_PI);

const double MU_0_BY_4_PI = MU_0 / (4.0 * M_PI);

const double PI_2 = M_PI / 2.0;

const double SQRT_EPS = sqrt(DBL_EPSILON);

double cel(double k_c, double p, double a, double b) {
if (k_c == 0.0) {
if (b != 0.0) {

return HUGE_VAL;
} else {
k_c = SQRT_EPS*SQRT_EPS;
}
} else {
k_c = fabs(k_c);
}

double m = 1.0;
double e = k_c;

double f, g;

if (p > 0.0) {
p = sqrt(p);
b = b / p;
} else {
double q;

f = k_c*k_c;
q = 1.0 - f;
g = 1.0 - p;
f -= p;
q *= b-a*p;
p = sqrt(f/g);
a = (a-b)/g;
b = -q/(g*g*p)+a*p;
}

while (1) {
f = a;
a += b/p;
g = e/p;
b += f*g;
b += b;
p += g;
g = m;
m += k_c;
if (fabs(g - k_c) > g * SQRT_EPS) {

k_c = sqrt(e);
k_c += k_c;
e = k_c*m;
} else {
break;
}
}

return PI_2 * (a*m+b) / (m*(m+p));
}

void compAdd(double contribution, double *compSum) {
double s   = compSum[0];
double cs  = compSum[1];

double t = s + contribution;
double c;
if (fabs(s) >= fabs(contribution)) {
c = (s - t) + contribution;
} else {
c = (contribution - t) + s;
}
compSum[0] = t;

double t2 = cs + c;
double cc;
if (fabs(cs) >= fabs(c)) {
cc = (cs - t2) + c;
} else {
cc = (c - t2) + cs;
}
compSum[1] = t2;
compSum[2] += cc;
}

double sws_A_z_ax_f(double zP) {
return atanh(1 / (fabs(zP) + fabs(1 - zP)));
}

double sws_A_z_ax_n(double zP) {

return copysign(1.0, zP) * log(zP / (zP - 1)) / 2;
}

double sws_A_z_ax(double zP) {
if (zP < -1 || zP >= 2) {
return sws_A_z_ax_f(zP);
} else {
return sws_A_z_ax_n(zP);
}
}

double sws_A_z_rad_f(double rhoP) {
return atanh(1 / (rhoP + hypot(rhoP, 1)));
}

double sws_A_z_rad_n(double rhoP) {
double cat = 1 / hypot(rhoP, 1);
double sat = sin(atan(rhoP) / 2);
double rc = rhoP * cat;
double num = rc + 1 + cat;
double den = rc + 2 * sat * sat;
return log(num / den) / 2;
}

double sws_A_z_rad(double rhoP) {
if (rhoP > 1) {
return sws_A_z_rad_f(rhoP);
} else {
return sws_A_z_rad_n(rhoP);
}
}

double sws_A_z_f(double rhoP, double zP) {
double r_i = hypot(rhoP, zP);
double r_f = hypot(rhoP, 1 - zP);
return atanh(1 / (r_i + r_f));
}

double sws_A_z_n(double rhoP, double zP) {
double omz = 1 - zP;

double r_i = hypot(rhoP, zP);
double r_f = hypot(rhoP, omz);

double alpha = atan2(rhoP, zP);
double sinAlphaHalf = sin(alpha / 2);

double beta = atan2(rhoP, omz);
double sinBetaHalf = sin(beta / 2);

double Ri_zP    = r_i * sinAlphaHalf * sinAlphaHalf;
double Rf_p_zM1 = r_f * sinBetaHalf  * sinBetaHalf;

double n = Ri_zP + Rf_p_zM1;

return (log(1 + n) - log(n)) / 2;
}

double sws_B_phi_rad(double rhoP) {
return 1 / (rhoP * hypot(rhoP, 1));
}

double sws_B_phi_f(double rhoP, double zP) {
double omz = 1 - zP;

double r_i = hypot(rhoP, zP);
double r_f = hypot(rhoP, omz);

double num = rhoP * (1/r_i + 1/r_f);
double den = rhoP * rhoP - zP * omz + r_i * r_f;

return num / den;
}

double sws_B_phi_n(double rhoP, double zP) {
double omz = 1 - zP;

double r_i = hypot(rhoP, zP);
double r_f = hypot(rhoP, omz);

double num = rhoP * (1/r_i + 1/r_f);

double alpha = atan2(rhoP, zP);
double sinAlphaHalf = sin(alpha / 2);

double beta = atan2(rhoP, omz);
double sinBetaHalf = sin(beta / 2);

double rfb_omza = r_f * sinBetaHalf * sinBetaHalf + omz * sinAlphaHalf * sinAlphaHalf;

double den = rhoP * rhoP + 2 * r_i * rfb_omza;

return num / den;
}

double cwl_A_phi_f(double rhoP, double zP) {
double sqrt_kCSqNum = hypot(zP, 1 - rhoP);
double sqrt_kCSqDen = hypot(zP, 1 + rhoP);

double kC = sqrt_kCSqNum / sqrt_kCSqDen;
double kSq = 4 * rhoP / (sqrt_kCSqDen * sqrt_kCSqDen);

double kCp1 = 1 + kC;
double arg1 = 2 * sqrt(kC) / kCp1;
double arg2 = 2 / (kCp1 * kCp1 * kCp1);
double C = cel(arg1, 1, 0, arg2);

return kSq/sqrt_kCSqDen * C;
}

double cwl_A_phi_n(double rhoP, double zP) {
double rhoP_m_1 = rhoP - 1;

double n = zP / rhoP_m_1;
double m = 1 + 2 / rhoP_m_1;

double num = n * n + 1;
double den = n * n + m * m;

double kCSq = num / den;

double prefac = 1 / (fabs(rhoP - 1) * sqrt(den));
double celPart = cel(sqrt(kCSq), 1, -1, 1);
return prefac * celPart;
}

double cwl_A_phi_v(double zP) {
double absZp = fabs(zP);

double kCInv = sqrt(4 + zP * zP) / absZp;

return cel(kCInv, 1, 1, -1) / absZp;
}

double cwl_B_rho_f(double rhoP, double zP) {
double sqrt_kCSqNum = hypot(zP, 1 - rhoP);
double sqrt_kCSqDen = hypot(zP, 1 + rhoP);

double kCSqNum = sqrt_kCSqNum * sqrt_kCSqNum;
double kCSqDen = sqrt_kCSqDen * sqrt_kCSqDen;

double kCSq = kCSqNum / kCSqDen;
double kC = sqrt(kCSq);

double D = cel(kC, 1, 0, 1);

double kCp1 = 1 + kC;
double arg1 = 2 * sqrt(kC) / kCp1;
double arg2 = 2 / (kCp1 * kCp1 * kCp1);
double C = cel(arg1, 1, 0, arg2);

double prefac = 4 * rhoP / (kCSqDen * sqrt_kCSqDen * kCSqNum);

return prefac * zP * (D - C);
}

double cwl_B_rho_n(double rhoP, double zP) {
double rhoP_m_1 = rhoP - 1;
double rd2 = rhoP_m_1 * rhoP_m_1;

double n = zP / rhoP_m_1;
double m = 1 + 2 / rhoP_m_1;

double sqrt_kCSqNum = hypot(n, 1);
double sqrt_kCSqDen = hypot(n, m);

double kCSqNum = sqrt_kCSqNum * sqrt_kCSqNum;
double kCSqDen = sqrt_kCSqDen * sqrt_kCSqDen;

double kC = sqrt_kCSqNum / sqrt_kCSqDen;

double D = cel(kC, 1, 0, 1);

double kCp1 = 1 + kC;
double arg1 = 2 * sqrt(kC) / kCp1;
double arg2 = 2 / (kCp1 * kCp1 * kCp1);
double C = arg2 * cel(arg1, 1, 0, 1);

double zP_rd5 = zP / (fabs(rhoP_m_1) * rd2 * rd2);

double prefac = 4 * rhoP / (kCSqDen * sqrt_kCSqDen * kCSqNum);

return prefac * zP_rd5 * (D - C);
}

double cwl_B_rho_v(double zP) {
double zPSq = zP * zP;

double kCSq = 1 / (1 + 4 / zPSq);
double kC = sqrt(kCSq);

double K = cel(kC, 1, 1, 1);
double E = cel(kC, 1, 1, kCSq);

return copysign(kC / 2 * ((2 / zPSq + 1) * E - K), zP);
}

double cwl_B_z_f1(double rhoP, double zP) {
double sqrt_kCSqNum = hypot(zP, 1 - rhoP);
double sqrt_kCSqDen = hypot(zP, 1 + rhoP);

double kC = sqrt_kCSqNum / sqrt_kCSqDen;

double K = cel(kC, 1, 1, 1);
double E = cel(kC, 1, 1, kC * kC);
double D = cel(kC, 1, 0, 1);

double prefac = 1 / (sqrt_kCSqDen * sqrt_kCSqNum * sqrt_kCSqNum);
double comb = (E - 2 * K + 2 * D);

return prefac * (E + rhoP * comb);
}

double cwl_B_z_f2(double rhoP, double zP) {
double sqrt_kCSqNum = hypot(zP, 1 - rhoP);
double sqrt_kCSqDen = hypot(zP, 1 + rhoP);

double kC = sqrt_kCSqNum / sqrt_kCSqDen;
double kCSq = kC * kC;

double zPSqP1 = zP * zP + 1;
double rhoPSq = rhoP * rhoP;
double t1 = zPSqP1 / rhoPSq + 1;
double t2 = 2 / rhoP;

double a = t1 + t2;
double b = t1 - t2;

double prefac = 1 / (sqrt(a) * b * rhoPSq * rhoP);

double cdScale = 1 + (2 + zPSqP1 / rhoP) / rhoP;

double E = cel(kC, 1, 1, kCSq);
double D = cel(kC, 1, 0, 1);

double kCP1 = 1 + kC;
double arg1 = 2 * sqrt(kC) / kCP1;
double arg2 = 2 / (kCP1 * kCP1 * kCP1);
double C = arg2 * cel(arg1, 1, 0, 1);

return prefac * (E + 4 * (C - D) / cdScale);
}

double cwl_B_z_n(double rhoP, double zP) {
double rp1 = rhoP - 1;

double n = zP / rp1;
double m = 1 + 2 / rp1;

double sqrt_kCSqNum = hypot(n, 1);
double sqrt_kCSqDen = hypot(n, m);

double kCSqDen = sqrt_kCSqDen * sqrt_kCSqDen;

double kC = sqrt_kCSqNum / sqrt_kCSqDen;

double prefac = 1 / (fabs(rp1) * rp1 * rp1 * kCSqDen * sqrt_kCSqDen);

return prefac * cel(kC, kC * kC, 1 + rhoP, 1 - rhoP);
}

double cwl_B_z_v(double zP) {
double kCSq = zP * zP / (4 + zP * zP);
double kC = sqrt(kCSq);

double f = zP * zP + 4;
double prefac = 1 / (f * sqrt(f));

return prefac * cel(kC, kCSq, 2, 0);
}

double straightWireSegment_A_z(double rhoP, double zP) {
if (rhoP == 0.0) {
if (zP < 0 || zP > 1) {
return sws_A_z_ax(zP);
} else {
fprintf(stderr, "evaluation locations on the wire segment (rho'=%g z'=%g) are not allowed\n", rhoP, zP);
return 0.0;
}
} else if (zP == 0.0 || zP == 1.0) {
return sws_A_z_rad(rhoP);
} else if (rhoP >= 1.0 || zP <= -1.0 || zP > 2.0) {
return sws_A_z_f(rhoP, zP);
} else {
return sws_A_z_n(rhoP, zP);
}
}

double straightWireSegment_B_phi(double rhoP, double zP) {
if (rhoP == 0.0) {
if (zP < 0 || zP > 1) {
return 0.0;
} else {
fprintf(stderr, "evaluation locations on the wire segment (rho'=%g z'=%g) are not allowed\n", rhoP, zP);
return 0.0;
}
} else if (zP == 0.0 || zP == 1.0) {
return sws_B_phi_rad(rhoP);
} else if (rhoP >= zP || rhoP >= 1 - zP || zP < 0.0 || zP > 1.0) {
return sws_B_phi_f(rhoP, zP);
} else {
return sws_B_phi_n(rhoP, zP);
}
}

double circularWireLoop_A_phi(double rhoP, double zP) {
if (rhoP == 0.0) {
return 0.0;
} else if (rhoP < 0.5 || rhoP > 2.0 || fabs(zP) >= 1.0) {
return cwl_A_phi_f(rhoP, zP);
} else if (rhoP != 1.0) {
return cwl_A_phi_n(rhoP, zP);
} else {
if (zP != 0) {
return cwl_A_phi_v(zP);
} else {
fprintf(stderr, "evaluation at location of wire loop (rho' = 1, z' = 0) is not defined\n");
return 0.0;
}
}
}

double circularWireLoop_B_rho(double rhoP, double zP) {
if (rhoP == 0.0 || zP == 0.0) {
if (rhoP != 1.0) {
return 0.0;
} else {
fprintf(stderr, "evaluation at location of wire loop (rho' = 1, z' = 0) is not defined\n");
return 0.0;
}
} else if (rhoP < 0.5 || rhoP > 2.0 || fabs(zP) >= 1.0) {
return cwl_B_rho_f(rhoP, zP);
} else if (rhoP != 1.0) {
return cwl_B_rho_n(rhoP, zP);
} else {
return cwl_B_rho_v(zP);
}
}

double circularWireLoop_B_z(double rhoP, double zP) {
if (rhoP < 0.5 || (rhoP <= 2 && fabs(zP) > 1)) {
return cwl_B_z_f1(rhoP, zP);
} else if (rhoP > 2) {
return cwl_B_z_f2(rhoP, zP);
} else if (rhoP != 1.0) {
return cwl_B_z_n(rhoP, zP);
} else {
if (zP != 0) {
return cwl_B_z_v(zP);
} else {
fprintf(stderr, "evaluation at location of wire loop (rho' = 1, z' = 0) is not defined\n");
return 0.0;
}
}
}

void vectorPotentialCircularFilament(double *center, double *normal, double radius,
double current, int nEvalPos, double *evalPos, double *vectorPotential) {

if (!isfinite(radius) || radius <= 0.0) {
printf("radius must be finite and positive, but is %g\n", radius);
return;
}

double aPrefactor = MU_0_BY_PI * current;

double nLen2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];

if (nLen2 == 0.0) {
printf("length of normal vector must not be zero");
return;
}

double nLen = sqrt(nLen2);

double eX = normal[0] / nLen;
double eY = normal[1] / nLen;
double eZ = normal[2] / nLen;

for (int idxEval = 0; idxEval < nEvalPos; ++idxEval) {

double r0x = evalPos[3 * idxEval + 0] - center[0];
double r0y = evalPos[3 * idxEval + 1] - center[1];
double r0z = evalPos[3 * idxEval + 2] - center[2];

double alignedZ = eX * r0x + eY * r0y + eZ * r0z;

double zP = alignedZ / radius;

double rParallelX = alignedZ * eX;
double rParallelY = alignedZ * eY;
double rParallelZ = alignedZ * eZ;

double rPerpX = r0x - rParallelX;
double rPerpY = r0y - rParallelY;
double rPerpZ = r0z - rParallelZ;

double alignedRSq = rPerpX * rPerpX + rPerpY * rPerpY + rPerpZ * rPerpZ;

if (alignedRSq > 0.0) {

double alignedR = sqrt(alignedRSq);

double eRX = rPerpX / alignedR;
double eRY = rPerpY / alignedR;
double eRZ = rPerpZ / alignedR;

double rhoP = alignedR / radius;

double aPhi = aPrefactor * circularWireLoop_A_phi(rhoP, zP);

double ePhiX = eRY * eZ - eRZ * eY;
double ePhiY = eRZ * eX - eRX * eZ;
double ePhiZ = eRX * eY - eRY * eX;

vectorPotential[3 * idxEval + 0] += aPhi * ePhiX;
vectorPotential[3 * idxEval + 1] += aPhi * ePhiY;
vectorPotential[3 * idxEval + 2] += aPhi * ePhiZ;
}
}
}

void magneticFieldCircularFilament(double *center, double *normal, double radius,
double current, int nEvalPos, double *evalPos, double *magneticField) {

if (!isfinite(radius) || radius <= 0.0) {
printf("radius must be finite and positive, but is %g\n", radius);
return;
}

double bPrefactor = MU_0_BY_PI * current / radius;

double nLen2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];

if (nLen2 == 0.0) {
printf("length of normal vector must not be zero");
return;
}

double nLen = sqrt(nLen2);

double eX = normal[0] / nLen;
double eY = normal[1] / nLen;
double eZ = normal[2] / nLen;

for (int idxEval = 0; idxEval < nEvalPos; ++idxEval) {

double r0x = evalPos[3 * idxEval + 0] - center[0];
double r0y = evalPos[3 * idxEval + 1] - center[1];
double r0z = evalPos[3 * idxEval + 2] - center[2];

double alignedZ = eX * r0x + eY * r0y + eZ * r0z;

double zP = alignedZ / radius;

double rParallelX = alignedZ * eX;
double rParallelY = alignedZ * eY;
double rParallelZ = alignedZ * eZ;

double rPerpX = r0x - rParallelX;
double rPerpY = r0y - rParallelY;
double rPerpZ = r0z - rParallelZ;

double alignedRSq = rPerpX * rPerpX + rPerpY * rPerpY + rPerpZ * rPerpZ;

double rhoP;
if (alignedRSq > 0.0) {

double alignedR = sqrt(alignedRSq);

double eRX = rPerpX / alignedR;
double eRY = rPerpY / alignedR;
double eRZ = rPerpZ / alignedR;

rhoP = alignedR / radius;

double bRho = bPrefactor * circularWireLoop_B_rho(rhoP, zP);

magneticField[3 * idxEval + 0] += bRho * eRX;
magneticField[3 * idxEval + 1] += bRho * eRY;
magneticField[3 * idxEval + 2] += bRho * eRZ;
} else {
rhoP = 0.0;
}

double bZ = bPrefactor * circularWireLoop_B_z(rhoP, zP);

magneticField[3 * idxEval + 0] += bZ * eX;
magneticField[3 * idxEval + 1] += bZ * eY;
magneticField[3 * idxEval + 2] += bZ * eZ;
}
}

void kernelVectorPotentialPolygonFilament(
double *vertices,
double current,
double *evalPos,
double *vectorPotential,
int idxSourceStart,
int idxSourceEnd,
int idxEvalStart,
int idxEvalEnd,
bool useCompensatedSummation) {

double aPrefactor = MU_0_BY_2_PI * current;

double *aXSum;
double *aYSum;
double *aZSum;
if (useCompensatedSummation) {
int numEvalPos = idxEvalEnd - idxEvalStart;

int numBytesToAllocate = 3 * numEvalPos * sizeof(double);

aXSum = (double *) malloc (numBytesToAllocate);
aYSum = (double *) malloc (numBytesToAllocate);
aZSum = (double *) malloc (numBytesToAllocate);

memset(aXSum, 0, numBytesToAllocate);
memset(aYSum, 0, numBytesToAllocate);
memset(aZSum, 0, numBytesToAllocate);
} else {
aXSum = NULL;
aYSum = NULL;
aZSum = NULL;
}

double x_i = vertices[idxSourceStart * 3 + 0];
double y_i = vertices[idxSourceStart * 3 + 1];
double z_i = vertices[idxSourceStart * 3 + 2];

for (int idxSource = idxSourceStart; idxSource < idxSourceEnd; ++idxSource) {

double x_f = vertices[(idxSource + 1) * 3 + 0];
double y_f = vertices[(idxSource + 1) * 3 + 1];
double z_f = vertices[(idxSource + 1) * 3 + 2];

double dx = x_f - x_i;
double dy = y_f - y_i;
double dz = z_f - z_i;

double l2 = dx * dx + dy * dy + dz * dz;
if (l2 == 0.0) {

continue;
}

double l = sqrt(l2);

double eX = dx / l;
double eY = dy / l;
double eZ = dz / l;

for (int idxEval = idxEvalStart; idxEval < idxEvalEnd; ++idxEval) {

double r0x = evalPos[3 * idxEval + 0] - x_i;
double r0y = evalPos[3 * idxEval + 1] - y_i;
double r0z = evalPos[3 * idxEval + 2] - z_i;

double alignedZ = eX * r0x + eY * r0y + eZ * r0z;

double zP = alignedZ / l;

double rPerpX = r0x - alignedZ * eX;
double rPerpY = r0y - alignedZ * eY;
double rPerpZ = r0z - alignedZ * eZ;

double alignedR = sqrt(rPerpX * rPerpX + rPerpY * rPerpY + rPerpZ * rPerpZ);

double rhoP = alignedR / l;

double aParallel = aPrefactor * straightWireSegment_A_z(rhoP, zP);

if (useCompensatedSummation) {
compAdd(aParallel * eX, &(aXSum[3 * (idxEval - idxEvalStart)]));
compAdd(aParallel * eY, &(aYSum[3 * (idxEval - idxEvalStart)]));
compAdd(aParallel * eZ, &(aZSum[3 * (idxEval - idxEvalStart)]));
} else {
vectorPotential[3 * idxEval + 0] += aParallel * eX;
vectorPotential[3 * idxEval + 1] += aParallel * eY;
vectorPotential[3 * idxEval + 2] += aParallel * eZ;
}
}

x_i = x_f;
y_i = y_f;
z_i = z_f;
}

if (useCompensatedSummation) {

for (int idxEval = idxEvalStart; idxEval < idxEvalEnd; ++idxEval) {
int relIdx = 3 * (idxEval - idxEvalStart);
vectorPotential[3 * idxEval + 0] += aXSum[relIdx + 0] + aXSum[relIdx + 1] + aXSum[relIdx + 2];
vectorPotential[3 * idxEval + 1] += aYSum[relIdx + 0] + aYSum[relIdx + 1] + aYSum[relIdx + 2];
vectorPotential[3 * idxEval + 2] += aZSum[relIdx + 0] + aZSum[relIdx + 1] + aZSum[relIdx + 2];
}

free(aXSum);
free(aYSum);
free(aZSum);
}
}

void kernelVectorPotentialPolygonFilament(
void (*vertexSupplier)(int i, double *point),
double current,
double *evalPos,
double *vectorPotential,
int idxSourceStart,
int idxSourceEnd,
int idxEvalStart,
int idxEvalEnd,
bool useCompensatedSummation) {

double aPrefactor = MU_0_BY_2_PI * current;

double *aXSum;
double *aYSum;
double *aZSum;
if (useCompensatedSummation) {
int numEvalPos = idxEvalEnd - idxEvalStart;

int numBytesToAllocate = 3 * numEvalPos * sizeof(double);

aXSum = (double *) malloc (numBytesToAllocate);
aYSum = (double *) malloc (numBytesToAllocate);
aZSum = (double *) malloc (numBytesToAllocate);

memset(aXSum, 0, numBytesToAllocate);
memset(aYSum, 0, numBytesToAllocate);
memset(aZSum, 0, numBytesToAllocate);
} else {
aXSum = NULL;
aYSum = NULL;
aZSum = NULL;
}

double pointData[3];
vertexSupplier(idxSourceStart, pointData);
double x_i = pointData[0];
double y_i = pointData[1];
double z_i = pointData[2];

for (int idxSource = idxSourceStart; idxSource < idxSourceEnd; ++idxSource) {

vertexSupplier(idxSource+1, pointData);
double x_f = pointData[0];
double y_f = pointData[1];
double z_f = pointData[2];

double dx = x_f - x_i;
double dy = y_f - y_i;
double dz = z_f - z_i;

double l2 = dx * dx + dy * dy + dz * dz;
if (l2 == 0.0) {

continue;
}

double l = sqrt(l2);

double eX = dx / l;
double eY = dy / l;
double eZ = dz / l;

for (int idxEval = idxEvalStart; idxEval < idxEvalEnd; ++idxEval) {

double r0x = evalPos[3 * idxEval + 0] - x_i;
double r0y = evalPos[3 * idxEval + 1] - y_i;
double r0z = evalPos[3 * idxEval + 2] - z_i;

double alignedZ = eX * r0x + eY * r0y + eZ * r0z;

double zP = alignedZ / l;

double rPerpX = r0x - alignedZ * eX;
double rPerpY = r0y - alignedZ * eY;
double rPerpZ = r0z - alignedZ * eZ;

double alignedR = sqrt(rPerpX * rPerpX + rPerpY * rPerpY + rPerpZ * rPerpZ);

double rhoP = alignedR / l;

double aParallel = aPrefactor * straightWireSegment_A_z(rhoP, zP);

if (useCompensatedSummation) {
compAdd(aParallel * eX, &(aXSum[3 * (idxEval - idxEvalStart)]));
compAdd(aParallel * eY, &(aYSum[3 * (idxEval - idxEvalStart)]));
compAdd(aParallel * eZ, &(aZSum[3 * (idxEval - idxEvalStart)]));
} else {
vectorPotential[3 * idxEval + 0] += aParallel * eX;
vectorPotential[3 * idxEval + 1] += aParallel * eY;
vectorPotential[3 * idxEval + 2] += aParallel * eZ;
}
}

x_i = x_f;
y_i = y_f;
z_i = z_f;
}

if (useCompensatedSummation) {

for (int idxEval = idxEvalStart; idxEval < idxEvalEnd; ++idxEval) {
int relIdx = 3 * (idxEval - idxEvalStart);
vectorPotential[3 * idxEval + 0] += aXSum[relIdx + 0] + aXSum[relIdx + 1] + aXSum[relIdx + 2];
vectorPotential[3 * idxEval + 1] += aYSum[relIdx + 0] + aYSum[relIdx + 1] + aYSum[relIdx + 2];
vectorPotential[3 * idxEval + 2] += aZSum[relIdx + 0] + aZSum[relIdx + 1] + aZSum[relIdx + 2];
}

free(aXSum);
free(aYSum);
free(aZSum);
}
}

void kernelMagneticFieldPolygonFilament(
double *vertices,
double current,
double *evalPos,
double *magneticField,
int idxSourceStart,
int idxSourceEnd,
int idxEvalStart,
int idxEvalEnd,
bool useCompensatedSummation) {

double bPrefactorL = MU_0_BY_4_PI * current;

double *bXSum;
double *bYSum;
double *bZSum;
if (useCompensatedSummation) {
int numEvalPos = idxEvalEnd - idxEvalStart;

int numBytesToAllocate = 3 * numEvalPos * sizeof(double);

bXSum = (double *) malloc (numBytesToAllocate);
bYSum = (double *) malloc (numBytesToAllocate);
bZSum = (double *) malloc (numBytesToAllocate);

memset(bXSum, 0, numBytesToAllocate);
memset(bYSum, 0, numBytesToAllocate);
memset(bZSum, 0, numBytesToAllocate);
} else {
bXSum = NULL;
bYSum = NULL;
bZSum = NULL;
}

double x_i = vertices[3 * idxSourceStart + 0];
double y_i = vertices[3 * idxSourceStart + 1];
double z_i = vertices[3 * idxSourceStart + 2];

for (int idxSource = idxSourceStart; idxSource < idxSourceEnd; ++idxSource) {

double x_f = vertices[3 * (idxSource + 1) + 0];
double y_f = vertices[3 * (idxSource + 1) + 1];
double z_f = vertices[3 * (idxSource + 1) + 2];

double dx = x_f - x_i;
double dy = y_f - y_i;
double dz = z_f - z_i;

double l2 = dx * dx + dy * dy + dz * dz;
if (l2 == 0.0) {

continue;
}

double l = sqrt(l2);

double bPrefactor = bPrefactorL / l;

double eX = dx / l;
double eY = dy / l;
double eZ = dz / l;

for (int idxEval = idxEvalStart; idxEval < idxEvalEnd; ++idxEval) {

double r0x = evalPos[3 * idxEval + 0] - x_i;
double r0y = evalPos[3 * idxEval + 1] - y_i;
double r0z = evalPos[3 * idxEval + 2] - z_i;

double alignedZ = eX * r0x + eY * r0y + eZ * r0z;

double rPerpX = r0x - alignedZ * eX;
double rPerpY = r0y - alignedZ * eY;
double rPerpZ = r0z - alignedZ * eZ;

double alignedRSq = rPerpX * rPerpX + rPerpY * rPerpY + rPerpZ * rPerpZ;

if (alignedRSq > 0.0) {

double alignedR = sqrt(alignedRSq);

double rhoP = alignedR / l;

double zP = alignedZ / l;

double bPhi = bPrefactor * straightWireSegment_B_phi(rhoP, zP);

double eRX = rPerpX / alignedR;
double eRY = rPerpY / alignedR;
double eRZ = rPerpZ / alignedR;

double ePhiX = eY * eRZ - eZ * eRY;
double ePhiY = eZ * eRX - eX * eRZ;
double ePhiZ = eX * eRY - eY * eRX;

if (useCompensatedSummation) {
compAdd(bPhi * ePhiX, &(bXSum[3 * (idxEval - idxEvalStart)]));
compAdd(bPhi * ePhiY, &(bYSum[3 * (idxEval - idxEvalStart)]));
compAdd(bPhi * ePhiZ, &(bZSum[3 * (idxEval - idxEvalStart)]));
} else {
magneticField[3 * idxEval + 0] += bPhi * ePhiX;
magneticField[3 * idxEval + 1] += bPhi * ePhiY;
magneticField[3 * idxEval + 2] += bPhi * ePhiZ;
}
}
}

x_i = x_f;
y_i = y_f;
z_i = z_f;
}

if (useCompensatedSummation) {

for (int idxEval = idxEvalStart; idxEval < idxEvalEnd; ++idxEval) {
int relIdx = 3 * (idxEval - idxEvalStart);
magneticField[3 * idxEval + 0] += bXSum[relIdx + 0] + bXSum[relIdx + 1] + bXSum[relIdx + 2];
magneticField[3 * idxEval + 1] += bYSum[relIdx + 0] + bYSum[relIdx + 1] + bYSum[relIdx + 2];
magneticField[3 * idxEval + 2] += bZSum[relIdx + 0] + bZSum[relIdx + 1] + bZSum[relIdx + 2];
}

free(bXSum);
free(bYSum);
free(bZSum);
}
}

void kernelMagneticFieldPolygonFilament(
void (*vertexSupplier)(int i, double *point),
double current,
double *evalPos,
double *magneticField,
int idxSourceStart,
int idxSourceEnd,
int idxEvalStart,
int idxEvalEnd,
bool useCompensatedSummation) {

double bPrefactorL = MU_0_BY_4_PI * current;

double *bXSum;
double *bYSum;
double *bZSum;
if (useCompensatedSummation) {
int numEvalPos = idxEvalEnd - idxEvalStart;

int numBytesToAllocate = 3 * numEvalPos * sizeof(double);

bXSum = (double *) malloc (numBytesToAllocate);
bYSum = (double *) malloc (numBytesToAllocate);
bZSum = (double *) malloc (numBytesToAllocate);

memset(bXSum, 0, numBytesToAllocate);
memset(bYSum, 0, numBytesToAllocate);
memset(bZSum, 0, numBytesToAllocate);
} else {
bXSum = NULL;
bYSum = NULL;
bZSum = NULL;
}

double pointData[3];
vertexSupplier(idxSourceStart, pointData);
double x_i = pointData[0];
double y_i = pointData[1];
double z_i = pointData[2];

for (int idxSource = idxSourceStart; idxSource < idxSourceEnd; ++idxSource) {

vertexSupplier(idxSource+1, pointData);
double x_f = pointData[0];
double y_f = pointData[1];
double z_f = pointData[2];

double dx = x_f - x_i;
double dy = y_f - y_i;
double dz = z_f - z_i;

double l2 = dx * dx + dy * dy + dz * dz;
if (l2 == 0.0) {

continue;
}

double l = sqrt(l2);

double bPrefactor = bPrefactorL / l;

double eX = dx / l;
double eY = dy / l;
double eZ = dz / l;

for (int idxEval = idxEvalStart; idxEval < idxEvalEnd; ++idxEval) {

double r0x = evalPos[3 * idxEval + 0] - x_i;
double r0y = evalPos[3 * idxEval + 1] - y_i;
double r0z = evalPos[3 * idxEval + 2] - z_i;

double alignedZ = eX * r0x + eY * r0y + eZ * r0z;

double rPerpX = r0x - alignedZ * eX;
double rPerpY = r0y - alignedZ * eY;
double rPerpZ = r0z - alignedZ * eZ;

double alignedRSq = rPerpX * rPerpX + rPerpY * rPerpY + rPerpZ * rPerpZ;

if (alignedRSq > 0.0) {

double alignedR = sqrt(alignedRSq);

double rhoP = alignedR / l;

double zP = alignedZ / l;

double bPhi = bPrefactor * straightWireSegment_B_phi(rhoP, zP);

double eRX = rPerpX / alignedR;
double eRY = rPerpY / alignedR;
double eRZ = rPerpZ / alignedR;

double ePhiX = eY * eRZ - eZ * eRY;
double ePhiY = eZ * eRX - eX * eRZ;
double ePhiZ = eX * eRY - eY * eRX;

if (useCompensatedSummation) {
compAdd(bPhi * ePhiX, &(bXSum[3 * (idxEval - idxEvalStart)]));
compAdd(bPhi * ePhiY, &(bYSum[3 * (idxEval - idxEvalStart)]));
compAdd(bPhi * ePhiZ, &(bZSum[3 * (idxEval - idxEvalStart)]));
} else {
magneticField[3 * idxEval + 0] += bPhi * ePhiX;
magneticField[3 * idxEval + 1] += bPhi * ePhiY;
magneticField[3 * idxEval + 2] += bPhi * ePhiZ;
}
}
}

x_i = x_f;
y_i = y_f;
z_i = z_f;
}

if (useCompensatedSummation) {

for (int idxEval = idxEvalStart; idxEval < idxEvalEnd; ++idxEval) {
int relIdx = 3 * (idxEval - idxEvalStart);
magneticField[3 * idxEval + 0] += bXSum[relIdx + 0] + bXSum[relIdx + 1] + bXSum[relIdx + 2];
magneticField[3 * idxEval + 1] += bYSum[relIdx + 0] + bYSum[relIdx + 1] + bYSum[relIdx + 2];
magneticField[3 * idxEval + 2] += bZSum[relIdx + 0] + bZSum[relIdx + 1] + bZSum[relIdx + 2];
}

free(bXSum);
free(bYSum);
free(bZSum);
}
}

void vectorPotentialPolygonFilament(
int numVertices,
double *vertices,
double current,
int numEvalPos,
double *evalPos,
double *vectorPotential,
int numProcessors,
bool useCompensatedSummation) {

if (numVertices < 2) {
printf("need at least 2 vertices, but only got %d\n", numVertices);
return;
}

int numSegments = numVertices - 1;

if (numProcessors < 1) {
printf("need at least 1 processor, but only got %d\n", numProcessors);
return;
}

if (current == 0.0) {
return;
}

if (numProcessors == 1) {

int idxSourceStart = 0;
int idxSourceEnd   = numSegments;
int idxEvalStart   = 0;
int idxEvalEnd     = numEvalPos;
kernelVectorPotentialPolygonFilament(
vertices, current,
evalPos,
vectorPotential,
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
} else {

if (numSegments > numEvalPos) {

int nThreads;
int nSourcePerThread;
int nSourceRemainder;
if (numSegments < numProcessors) {
nThreads = numSegments;

nSourcePerThread = 1;
nSourceRemainder = 0;
} else {
nThreads = numProcessors;

nSourcePerThread = numSegments / nThreads;
nSourceRemainder = numSegments % nThreads;
}

int idxEvalStart   = 0;
int idxEvalEnd     = numEvalPos;

int numBytes = nThreads * numEvalPos * 3 * sizeof(double);
double *vectorPotentialContributions = (double *) malloc(numBytes);
if (vectorPotentialContributions == NULL) {
printf("failed to allocate temporary array for vector potential contributions\n");
return;
}
memset(vectorPotentialContributions, 0, numBytes);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

int idxSourceStart =  idxThread      * nSourcePerThread;
int idxSourceEnd   = (idxThread + 1) * nSourcePerThread;
if (idxThread < nSourceRemainder) {
idxSourceStart += idxThread;
idxSourceEnd   += idxThread + 1;
} else {
idxSourceStart += nSourceRemainder;
idxSourceEnd   += nSourceRemainder;
}

kernelVectorPotentialPolygonFilament(
vertices, current,
evalPos,
&(vectorPotentialContributions[idxThread * numEvalPos * 3]),
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
}

if (useCompensatedSummation) {
for (int i = 0; i < numEvalPos; ++i) {
double sumX[3] = { 0.0, 0.0, 0.0 };
double sumY[3] = { 0.0, 0.0, 0.0 };
double sumZ[3] = { 0.0, 0.0, 0.0 };
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

compAdd(vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 0], sumX);
compAdd(vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 1], sumY);
compAdd(vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 2], sumZ);
}
vectorPotential[i * 3 + 0] += sumX[0] + sumX[1] + sumX[2];
vectorPotential[i * 3 + 1] += sumY[0] + sumY[1] + sumY[2];
vectorPotential[i * 3 + 2] += sumZ[0] + sumZ[1] + sumZ[2];
}
} else {
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {
for (int i = 0; i < numEvalPos; ++i) {
vectorPotential[i * 3 + 0] += vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 0];
vectorPotential[i * 3 + 1] += vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 1];
vectorPotential[i * 3 + 2] += vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 2];
}
}
}

free(vectorPotentialContributions);
} else {

int nThreads;
int idxSourceStart = 0;
int idxSourceEnd   = numSegments;
int nEvalPerThread;
int nEvalRemainder;
if (numEvalPos < numProcessors) {
nThreads = numEvalPos;

nEvalPerThread = 1;
nEvalRemainder = 0;
} else {
nThreads = numProcessors;

nEvalPerThread = numEvalPos / nThreads;
nEvalRemainder = numEvalPos % nThreads;
}

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

int idxEvalStart =  idxThread      * nEvalPerThread;
int idxEvalEnd   = (idxThread + 1) * nEvalPerThread;
if (idxThread < nEvalRemainder) {
idxEvalStart += idxThread;
idxEvalEnd   += idxThread + 1;
} else {
idxEvalStart += nEvalRemainder;
idxEvalEnd   += nEvalRemainder;
}

kernelVectorPotentialPolygonFilament(
vertices, current,
evalPos,
vectorPotential,
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
}
}
}
}

void vectorPotentialPolygonFilament(
int numVertices,
void (*vertexSupplier)(int i, double *point),
double current,
int numEvalPos,
double *evalPos,
double *vectorPotential,
int numProcessors,
bool useCompensatedSummation) {

if (numVertices < 2) {
printf("need at least 2 vertices, but only got %d\n", numVertices);
return;
}

int numSegments = numVertices - 1;

if (numProcessors < 1) {
printf("need at least 1 processor, but only got %d\n", numProcessors);
return;
}

if (current == 0.0) {
return;
}

if (numProcessors == 1) {

int idxSourceStart = 0;
int idxSourceEnd   = numSegments;
int idxEvalStart   = 0;
int idxEvalEnd     = numEvalPos;
kernelVectorPotentialPolygonFilament(
vertexSupplier, current,
evalPos,
vectorPotential,
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
} else {

if (numSegments > numEvalPos) {

int nThreads;
int nSourcePerThread;
int nSourceRemainder;
if (numSegments < numProcessors) {
nThreads = numSegments;

nSourcePerThread = 1;
nSourceRemainder = 0;
} else {
nThreads = numProcessors;

nSourcePerThread = numSegments / nThreads;
nSourceRemainder = numSegments % nThreads;
}

int idxEvalStart   = 0;
int idxEvalEnd     = numEvalPos;

int numBytes = nThreads * numEvalPos * 3 * sizeof(double);
double *vectorPotentialContributions = (double *) malloc(numBytes);
if (vectorPotentialContributions == NULL) {
printf("failed to allocate temporary array for vector potential contributions\n");
return;
}
memset(vectorPotentialContributions, 0, numBytes);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

int idxSourceStart =  idxThread      * nSourcePerThread;
int idxSourceEnd   = (idxThread + 1) * nSourcePerThread;
if (idxThread < nSourceRemainder) {
idxSourceStart += idxThread;
idxSourceEnd   += idxThread + 1;
} else {
idxSourceStart += nSourceRemainder;
idxSourceEnd   += nSourceRemainder;
}

kernelVectorPotentialPolygonFilament(
vertexSupplier, current,
evalPos,
&(vectorPotentialContributions[idxThread * numEvalPos * 3]),
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
}

if (useCompensatedSummation) {
for (int i=0; i<numEvalPos; ++i) {
double sumX[3] = { 0.0, 0.0, 0.0 };
double sumY[3] = { 0.0, 0.0, 0.0 };
double sumZ[3] = { 0.0, 0.0, 0.0 };
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

compAdd(vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 0], sumX);
compAdd(vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 1], sumY);
compAdd(vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 2], sumZ);
}
vectorPotential[i * 3 + 0] += sumX[0] + sumX[1] + sumX[2];
vectorPotential[i * 3 + 1] += sumY[0] + sumY[1] + sumY[2];
vectorPotential[i * 3 + 2] += sumZ[0] + sumZ[1] + sumZ[2];
}
} else {
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {
for (int i=0; i<numEvalPos; ++i) {
vectorPotential[i * 3 + 0] += vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 0];
vectorPotential[i * 3 + 1] += vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 1];
vectorPotential[i * 3 + 2] += vectorPotentialContributions[(idxThread * numEvalPos + i) * 3 + 2];
}
}
}

free(vectorPotentialContributions);
} else {

int nThreads;
int idxSourceStart = 0;
int idxSourceEnd   = numSegments;
int nEvalPerThread;
int nEvalRemainder;
if (numEvalPos < numProcessors) {
nThreads = numEvalPos;

nEvalPerThread = 1;
nEvalRemainder = 0;
} else {
nThreads = numProcessors;

nEvalPerThread = numEvalPos / nThreads;
nEvalRemainder = numEvalPos % nThreads;
}

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

int idxEvalStart =  idxThread      * nEvalPerThread;
int idxEvalEnd   = (idxThread + 1) * nEvalPerThread;
if (idxThread < nEvalRemainder) {
idxEvalStart += idxThread;
idxEvalEnd   += idxThread + 1;
} else {
idxEvalStart += nEvalRemainder;
idxEvalEnd   += nEvalRemainder;
}

kernelVectorPotentialPolygonFilament(
vertexSupplier, current,
evalPos,
vectorPotential,
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
}
}
}
}

void magneticFieldPolygonFilament(
int numVertices,
double *vertices,
double current,
int numEvalPos,
double *evalPos,
double *magneticField,
int numProcessors,
bool useCompensatedSummation) {

if (numVertices < 2) {
printf("need at least 2 vertices, but only got %d\n", numVertices);
return;
}

int numSegments = numVertices - 1;

if (numProcessors < 1) {
printf("need at least 1 processor, but only got %d\n", numProcessors);
return;
}

if (current == 0.0) {
return;
}

if (numProcessors == 1) {

int idxSourceStart = 0;
int idxSourceEnd   = numSegments;
int idxEvalStart   = 0;
int idxEvalEnd     = numEvalPos;
kernelMagneticFieldPolygonFilament(
vertices, current,
evalPos,
magneticField,
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
} else {

if (numSegments > numEvalPos) {

int nThreads;
int nSourcePerThread;
int nSourceRemainder;
if (numSegments < numProcessors) {
nThreads = numSegments;

nSourcePerThread = 1;
nSourceRemainder = 0;
} else {
nThreads = numProcessors;

nSourcePerThread = numSegments / nThreads;
nSourceRemainder = numSegments % nThreads;
}

int idxEvalStart   = 0;
int idxEvalEnd     = numEvalPos;

int numBytes = nThreads * numEvalPos * 3 * sizeof(double);
double *magneticFieldContributions = (double *) malloc(numBytes);
if (magneticFieldContributions == NULL) {
printf("failed to allocate temporary array for magnetic field contributions\n");
return;
}
memset(magneticFieldContributions, 0, numBytes);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

int idxSourceStart =  idxThread      * nSourcePerThread;
int idxSourceEnd   = (idxThread + 1) * nSourcePerThread;
if (idxThread < nSourceRemainder) {
idxSourceStart += idxThread;
idxSourceEnd   += idxThread + 1;
} else {
idxSourceStart += nSourceRemainder;
idxSourceEnd   += nSourceRemainder;
}

kernelMagneticFieldPolygonFilament(
vertices, current,
evalPos,
&(magneticFieldContributions[idxThread * numEvalPos * 3]),
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
}

if (useCompensatedSummation) {
for (int i=0; i<numEvalPos; ++i) {
double sumX[3] = { 0.0, 0.0, 0.0 };
double sumY[3] = { 0.0, 0.0, 0.0 };
double sumZ[3] = { 0.0, 0.0, 0.0 };
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

compAdd(magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 0], sumX);
compAdd(magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 1], sumY);
compAdd(magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 2], sumZ);
}
magneticField[i * 3 + 0] += sumX[0] + sumX[1] + sumX[2];
magneticField[i * 3 + 1] += sumY[0] + sumY[1] + sumY[2];
magneticField[i * 3 + 2] += sumZ[0] + sumZ[1] + sumZ[2];
}
} else {
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {
for (int i = 0; i < numEvalPos; ++i) {
magneticField[i * 3 + 0] += magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 0];
magneticField[i * 3 + 1] += magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 1];
magneticField[i * 3 + 2] += magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 2];
}
}
}

free(magneticFieldContributions);
} else {

int nThreads;
int idxSourceStart = 0;
int idxSourceEnd   = numSegments;
int nEvalPerThread;
int nEvalRemainder;
if (numEvalPos < numProcessors) {
nThreads = numEvalPos;

nEvalPerThread = 1;
nEvalRemainder = 0;
} else {
nThreads = numProcessors;

nEvalPerThread = numEvalPos / nThreads;
nEvalRemainder = numEvalPos % nThreads;
}

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

int idxEvalStart =  idxThread      * nEvalPerThread;
int idxEvalEnd   = (idxThread + 1) * nEvalPerThread;
if (idxThread < nEvalRemainder) {
idxEvalStart += idxThread;
idxEvalEnd   += idxThread + 1;
} else {
idxEvalStart += nEvalRemainder;
idxEvalEnd   += nEvalRemainder;
}

kernelMagneticFieldPolygonFilament(
vertices, current,
evalPos,
magneticField,
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
}
}
}
}

void magneticFieldPolygonFilament(
int numVertices,
void (*vertexSupplier)(int i, double *point),
double current,
int numEvalPos,
double *evalPos,
double *magneticField,
int numProcessors,
bool useCompensatedSummation) {

if (numVertices < 2) {
printf("need at least 2 vertices, but only got %d\n", numVertices);
return;
}

int numSegments = numVertices - 1;

if (numProcessors < 1) {
printf("need at least 1 processor, but only got %d\n", numProcessors);
return;
}

if (current == 0.0) {
return;
}

if (numProcessors == 1) {

int idxSourceStart = 0;
int idxSourceEnd   = numSegments;
int idxEvalStart   = 0;
int idxEvalEnd     = numEvalPos;
kernelMagneticFieldPolygonFilament(
vertexSupplier, current,
evalPos,
magneticField,
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
} else {

if (numSegments > numEvalPos) {

int nThreads;
int nSourcePerThread;
int nSourceRemainder;
if (numSegments < numProcessors) {
nThreads = numSegments;

nSourcePerThread = 1;
nSourceRemainder = 0;
} else {
nThreads = numProcessors;

nSourcePerThread = numSegments / nThreads;
nSourceRemainder = numSegments % nThreads;
}

int idxEvalStart   = 0;
int idxEvalEnd     = numEvalPos;

int numBytes = nThreads * numEvalPos * 3 * sizeof(double);
double *magneticFieldContributions = (double *) malloc(numBytes);
if (magneticFieldContributions == NULL) {
printf("failed to allocate temporary array for magnetic field contributions\n");
return;
}
memset(magneticFieldContributions, 0, numBytes);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

int idxSourceStart =  idxThread      * nSourcePerThread;
int idxSourceEnd   = (idxThread + 1) * nSourcePerThread;
if (idxThread < nSourceRemainder) {
idxSourceStart += idxThread;
idxSourceEnd   += idxThread + 1;
} else {
idxSourceStart += nSourceRemainder;
idxSourceEnd   += nSourceRemainder;
}

kernelMagneticFieldPolygonFilament(
vertexSupplier, current,
evalPos,
&(magneticFieldContributions[idxThread * numEvalPos * 3]),
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
}

if (useCompensatedSummation) {
for (int i = 0; i < numEvalPos; ++i) {
double sumX[3] = { 0.0, 0.0, 0.0 };
double sumY[3] = { 0.0, 0.0, 0.0 };
double sumZ[3] = { 0.0, 0.0, 0.0 };
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

compAdd(magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 0], sumX);
compAdd(magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 1], sumY);
compAdd(magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 2], sumZ);
}
magneticField[i * 3 + 0] += sumX[0] + sumX[1] + sumX[2];
magneticField[i * 3 + 1] += sumY[0] + sumY[1] + sumY[2];
magneticField[i * 3 + 2] += sumZ[0] + sumZ[1] + sumZ[2];
}
} else {
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {
for (int i = 0; i < numEvalPos; ++i) {
magneticField[i * 3 + 0] += magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 0];
magneticField[i * 3 + 1] += magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 1];
magneticField[i * 3 + 2] += magneticFieldContributions[(idxThread * numEvalPos + i) * 3 + 2];
}
}
}

free(magneticFieldContributions);
} else {

int nThreads;
int idxSourceStart = 0;
int idxSourceEnd   = numSegments;
int nEvalPerThread;
int nEvalRemainder;
if (numEvalPos < numProcessors) {
nThreads = numEvalPos;

nEvalPerThread = 1;
nEvalRemainder = 0;
} else {
nThreads = numProcessors;

nEvalPerThread = numEvalPos / nThreads;
nEvalRemainder = numEvalPos % nThreads;
}

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int idxThread = 0; idxThread < nThreads; ++idxThread) {

int idxEvalStart =  idxThread      * nEvalPerThread;
int idxEvalEnd   = (idxThread + 1) * nEvalPerThread;
if (idxThread < nEvalRemainder) {
idxEvalStart += idxThread;
idxEvalEnd   += idxThread + 1;
} else {
idxEvalStart += nEvalRemainder;
idxEvalEnd   += nEvalRemainder;
}

kernelMagneticFieldPolygonFilament(
vertexSupplier, current,
evalPos,
magneticField,
idxSourceStart, idxSourceEnd, idxEvalStart, idxEvalEnd,
useCompensatedSummation);
}
}
}
}

}

// source: util/file_io/file_io.cc

// header: util/file_io/file_io.h

#ifndef UTIL_FILE_IO_FILE_IO_H_
#define UTIL_FILE_IO_FILE_IO_H_

#include <filesystem>
#include <string>

#include "absl/status/statusor.h"

namespace file_io {

absl::StatusOr<std::string> ReadFile(const std::filesystem::path& filename);

absl::Status WriteFile(const std::filesystem::path& filename,
const std::string& contents);

}

#endif

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace fs = std::filesystem;

absl::StatusOr<std::string> file_io::ReadFile(const fs::path& filename) {
if (!fs::exists(filename)) {
return absl::NotFoundError("File " + filename.string() + " not found.");
}

std::ifstream ifs(filename.c_str());
if (!ifs.is_open()) {
return absl::PermissionDeniedError("File " + filename.string() +
" could not be opened for reading.");
}

std::string contents(std::istreambuf_iterator<char>{ifs}, {});
return contents;
}

absl::Status file_io::WriteFile(const std::filesystem::path& filename,
const std::string& contents) {
std::ofstream file_stream(filename.string());
if (!file_stream.is_open()) {
return absl::PermissionDeniedError("File " + filename.string() +
" could not be opened for writing.");
}

file_stream << contents;

file_stream.flush();
file_stream.close();

return absl::OkStatus();
}

// source: util/hdf5_io/hdf5_io.cc

// header: util/hdf5_io/hdf5_io.h

#ifndef UTIL_HDF5_IO_HDF5_IO_H_
#define UTIL_HDF5_IO_HDF5_IO_H_

#include <Eigen/Dense>
#include <string>
#include <type_traits>
#include <vector>

#include "H5Cpp.h"

namespace hdf5_io {

using RowMatrixXd =
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void WriteH5Dataset(const RowMatrixXd& m, const std::string& name,
H5::H5File& file);

template <typename Vector>
requires(Vector::ColsAtCompileTime == 1) ||
std::is_same_v<Vector, std::vector<double>> ||
std::is_same_v<Vector, std::vector<int>>
void WriteH5Dataset(const Vector& v, const std::string& name,
H5::H5File& file) {
const hsize_t size = v.size();
H5::DataSpace dataspace( 1,  &size);

H5::PredType type = H5::PredType::NATIVE_DOUBLE;
if constexpr (std::is_integral_v<typename Vector::value_type>) {
type = H5::PredType::NATIVE_INT;
}

H5::DataSet dataset = file.createDataSet(name, type, dataspace);
if constexpr (std::is_convertible_v<Vector, Eigen::VectorXd> ||
std::is_convertible_v<Vector, Eigen::VectorXi>) {
dataset.write(v.eval().data(), type);
} else {
dataset.write(v.data(), type);
}
}

extern template void WriteH5Dataset(const Eigen::VectorXd& v,
const std::string& name, H5::H5File& file);
extern template void WriteH5Dataset(const Eigen::VectorXi& v,
const std::string& name, H5::H5File& file);

void WriteH5Dataset(const std::vector<std::string>& vs, const std::string& name,
H5::H5File& file);

void WriteH5Dataset(const std::string& str, const std::string& name,
H5::H5File& file);

template <typename Scalar>
requires std::is_scalar_v<Scalar>
void WriteH5Dataset(Scalar v, const std::string& name, H5::H5File& file) {
H5::DataSpace dataspace(H5S_SCALAR);

H5::PredType type = H5::PredType::NATIVE_DOUBLE;
if constexpr (std::is_same_v<Scalar, int>) {
type = H5::PredType::NATIVE_INT;
} else if (std::is_same_v<Scalar, bool>) {
type = H5::PredType::NATIVE_HBOOL;
}

H5::DataSet dataset = file.createDataSet(name, type, dataspace);
dataset.write(&v, type);
}

void ReadH5Dataset(RowMatrixXd& m, const std::string& dataset,
H5::H5File& file);

template <typename Vector>
requires std::is_same_v<Vector, Eigen::VectorXd> ||
std::is_same_v<Vector, Eigen::VectorXi> ||
std::is_same_v<Vector, std::vector<double>> ||
std::is_same_v<Vector, std::vector<int>>
void ReadH5Dataset(Vector& v, const std::string& dataset, H5::H5File& file) {
H5::DataSet ds = file.openDataSet(dataset);

H5::DataSpace space = ds.getSpace();
hsize_t size;
space.getSimpleExtentDims(&size);

H5::PredType type = H5::PredType::NATIVE_DOUBLE;
if constexpr (std::is_same_v<Vector, Eigen::VectorXi> ||
std::is_same_v<Vector, std::vector<int>>) {
type = H5::PredType::NATIVE_INT;
}

v.resize(static_cast<Eigen::Index>(size));
ds.read(v.data(), type);
}

void ReadH5Dataset(std::vector<std::string>& vs, const std::string& dataset,
H5::H5File& file);

void ReadH5Dataset(std::string& str, const std::string& dataset,
H5::H5File& file);

template <typename Scalar>
requires std::is_same_v<Scalar, double> || std::is_same_v<Scalar, int> ||
std::is_same_v<Scalar, bool>
void ReadH5Dataset(Scalar& v, const std::string& dataset, H5::H5File& file) {
H5::DataSet ds = file.openDataSet(dataset);

H5::PredType type = H5::PredType::NATIVE_DOUBLE;
if constexpr (std::is_same_v<Scalar, int>) {
type = H5::PredType::NATIVE_INT;
} else if (std::is_same_v<Scalar, bool>) {
type = H5::PredType::NATIVE_HBOOL;
}

ds.read(&v, type);
}

extern template void ReadH5Dataset(double& v, const std::string& dataset,
H5::H5File& file);
extern template void ReadH5Dataset(int& v, const std::string& dataset,
H5::H5File& file);
extern template void ReadH5Dataset(bool& v, const std::string& dataset,
H5::H5File& file);

int GetRank(const H5::DataSet& dataset);

std::vector<hsize_t> GetExtent(const H5::DataSet& dataset);

template <typename Vector>
requires std::is_same_v<Vector, Eigen::VectorXd> ||
std::is_same_v<Vector, Eigen::VectorXi> ||
std::is_same_v<Vector, std::vector<double>> ||
std::is_same_v<Vector, std::vector<int>>
void ReadH5Dataset(Vector& m_vector, std::vector<hsize_t>& m_dims,
const std::string& name, H5::H5File& file) {
H5::DataSet dataset = file.openDataSet(name);
m_dims = GetExtent(dataset);

hsize_t total_size = 1;
for (auto dim : m_dims) {
total_size *= dim;
}

m_vector.resize(static_cast<Eigen::Index>(total_size));

H5::PredType type = H5::PredType::NATIVE_DOUBLE;
if constexpr (std::is_integral_v<typename Vector::value_type>) {
type = H5::PredType::NATIVE_INT;
}

dataset.read(m_vector.data(), type);
}

template <typename Vector>
requires(Vector::ColsAtCompileTime == 1) ||
std::is_same_v<Vector, std::vector<double>> ||
std::is_same_v<Vector, std::vector<int>>
void WriteH5Dataset(const Vector& vector, const std::vector<hsize_t>& dims,
const std::string& name, H5::H5File& h5file_output) {
H5::DataSpace dataspace( dims.size(),  dims.data());

H5::PredType type = H5::PredType::NATIVE_DOUBLE;
if constexpr (std::is_integral_v<typename Vector::value_type>) {
type = H5::PredType::NATIVE_INT;
}
H5::DataSet dataset = h5file_output.createDataSet(name, type, dataspace);

dataset.write(vector.data(), type);
}

}
#endif

#include <array>
#include <string>
#include <vector>

void hdf5_io::WriteH5Dataset(const hdf5_io::RowMatrixXd& m,
const std::string& name, H5::H5File& file) {
const std::array<hsize_t, 2> dims = {static_cast<hsize_t>(m.rows()),
static_cast<hsize_t>(m.cols())};
H5::DataSpace dataspace( 2,  dims.data());
H5::DataSet dataset =
file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
dataset.write(m.data(), H5::PredType::NATIVE_DOUBLE);
}

template void hdf5_io::WriteH5Dataset(const Eigen::VectorXd& v,
const std::string& name,
H5::H5File& file);

template void hdf5_io::WriteH5Dataset(const Eigen::VectorXi& v,
const std::string& name,
H5::H5File& file);

void hdf5_io::WriteH5Dataset(const std::vector<std::string>& vs,
const std::string& name, H5::H5File& file) {
const hsize_t size = vs.size();
H5::DataSpace dataspace( 1,  &size);
H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
H5::DataSet dataset = file.createDataSet(name, str_type, dataspace);

std::vector<const char*> pointers;
pointers.reserve(vs.size());
for (const std::string& s : vs) {
pointers.push_back(s.c_str());
}
dataset.write(pointers.data(), str_type);
}

void hdf5_io::WriteH5Dataset(const std::string& str, const std::string& name,
H5::H5File& file) {
H5::DataSpace dataspace(H5S_SCALAR);
H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
H5::DataSet dataset = file.createDataSet(name, str_type, dataspace);
dataset.write(str, str_type);
}

void hdf5_io::ReadH5Dataset(hdf5_io::RowMatrixXd& m, const std::string& dataset,
H5::H5File& file) {
H5::DataSet ds = file.openDataSet(dataset);

H5::DataSpace space = ds.getSpace();
std::array<hsize_t, 2> dims;
space.getSimpleExtentDims(dims.data());

m.resize(static_cast<Eigen::Index>(dims[0]),
static_cast<Eigen::Index>(dims[1]));
ds.read(m.data(), H5::PredType::NATIVE_DOUBLE);
}

void hdf5_io::ReadH5Dataset(std::vector<std::string>& vs,
const std::string& dataset, H5::H5File& file) {
H5::DataSet ds = file.openDataSet(dataset);

H5::DataSpace space = ds.getSpace();
hsize_t size;
space.getSimpleExtentDims(&size);

H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);

std::vector<char*> pointers(size, nullptr);
ds.read(pointers.data(), str_type);

vs.resize(size);
for (hsize_t i = 0; i < size; ++i) {
vs[i] = pointers[i] == nullptr ? std::string() : std::string(pointers[i]);
}
if (size > 0) {
H5::DataSpace space_for_reclaim = ds.getSpace();
H5::DataSet::vlenReclaim(str_type, space_for_reclaim,
H5::DSetMemXferPropList::DEFAULT, pointers.data());
}
}

void hdf5_io::ReadH5Dataset(std::string& str, const std::string& dataset,
H5::H5File& file) {
H5::DataSet ds = file.openDataSet(dataset);
H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
ds.read(str, str_type);
}

template void hdf5_io::ReadH5Dataset(double& v, const std::string& dataset,
H5::H5File& file);
template void hdf5_io::ReadH5Dataset(int& v, const std::string& dataset,
H5::H5File& file);
template void hdf5_io::ReadH5Dataset(bool& v, const std::string& dataset,
H5::H5File& file);

int hdf5_io::GetRank(const H5::DataSet& dataset) {
H5::DataSpace dataspace = dataset.getSpace();
return dataspace.getSimpleExtentNdims();
}

std::vector<hsize_t> hdf5_io::GetExtent(const H5::DataSet& dataset) {
H5::DataSpace dataspace = dataset.getSpace();
int rank = dataspace.getSimpleExtentNdims();
std::vector<hsize_t> extent(rank);
dataspace.getSimpleExtentDims(extent.data(), NULL);
return extent;
}

// source: util/json_io/json_io.cc

// header: util/json_io/json_io.h

#ifndef UTIL_JSON_IO_JSON_IO_H_
#define UTIL_JSON_IO_JSON_IO_H_

#include <Eigen/Dense>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"

namespace json_io {

absl::StatusOr<nlohmann::json> JsonParse(const std::string& json_text);

absl::StatusOr<std::optional<bool>> JsonReadBool(const nlohmann::json& j,
const std::string& name);

absl::StatusOr<std::optional<int>> JsonReadInt(const nlohmann::json& j,
const std::string& name);

absl::StatusOr<std::optional<double>> JsonReadDouble(const nlohmann::json& j,
const std::string& name);

absl::StatusOr<std::optional<std::string>> JsonReadString(
const nlohmann::json& j, const std::string& name);

absl::StatusOr<std::optional<Eigen::VectorXi>> JsonReadVectorInt(
const nlohmann::json& j, const std::string& name);

absl::StatusOr<std::optional<Eigen::VectorXd>> JsonReadVectorDouble(
const nlohmann::json& j, const std::string& name);

}

#endif

#include <string>
#include <vector>

#include "absl/strings/str_format.h"

namespace {
using nlohmann::json;
}

namespace json_io {

absl::StatusOr<json> JsonParse(const std::string& json_text) {
try {
return json::parse(json_text);
} catch (const json::parse_error& parse_error) {
return absl::InvalidArgumentError(
absl::StrFormat("input is not valid JSON: %s", parse_error.what()));
}
}

absl::StatusOr<std::optional<bool>> JsonReadBool(const json& j,
const std::string& name) {
if (!j.contains(name)) {

return std::nullopt;
}

if (!j[name].is_boolean()) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON element '%s' is not a boolean", name));
}

return j[name];
}

absl::StatusOr<std::optional<int>> JsonReadInt(const json& j,
const std::string& name) {
if (!j.contains(name)) {

return std::nullopt;
}

if (!j[name].is_number_integer()) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON element '%s' is not an integer", name));
}

return j[name];
}

absl::StatusOr<std::optional<double>> JsonReadDouble(const json& j,
const std::string& name) {
if (!j.contains(name)) {

return std::nullopt;
}

if (!j[name].is_number()) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON element '%s' is not a number", name));
}

return j[name];
}

absl::StatusOr<std::optional<std::string>> JsonReadString(
const json& j, const std::string& name) {
if (!j.contains(name)) {

return std::nullopt;
}

if (!j[name].is_string()) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON element '%s' is not a string", name));
}

return j[name];
}

absl::StatusOr<std::optional<Eigen::VectorXi>> JsonReadVectorInt(
const json& j, const std::string& name) {
if (!j.contains(name)) {

return std::nullopt;
}

if (!j[name].is_array()) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON element '%s' is not an array", name));
}

std::vector<int> entries;
int i = 0;
for (const auto& entry : j[name]) {
if (entry.is_number_integer()) {
entries.push_back(entry);
} else {
return absl::InvalidArgumentError(
absl::StrFormat("JSON entry '%s'[%d] is not an integer", name, i));
}
i++;
}

Eigen::VectorXi::Index entries_size =
static_cast<Eigen::VectorXi::Index>(entries.size());
Eigen::VectorXi entries_vector(entries_size);
for (Eigen::VectorXi::Index i = 0; i < entries_size; ++i) {
entries_vector[i] = entries[i];
}

return entries_vector;
}

absl::StatusOr<std::optional<Eigen::VectorXd>> JsonReadVectorDouble(
const json& j, const std::string& name) {
if (!j.contains(name)) {

return std::nullopt;
}

if (!j[name].is_array()) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON element '%s' is not an array", name));
}

std::vector<double> entries;
int i = 0;
for (const auto& entry : j[name]) {
if (entry.is_number()) {
entries.push_back(entry);
} else {
return absl::InvalidArgumentError(
absl::StrFormat("JSON entry '%s'[%d] is not a number", name, i));
}
i++;
}

Eigen::VectorXd::Index entries_size =
static_cast<Eigen::VectorXd::Index>(entries.size());
Eigen::VectorXd entries_vector(entries_size);
for (Eigen::VectorXd::Index i = 0; i < entries_size; ++i) {
entries_vector[i] = entries[i];
}

return entries_vector;
}

}

// source: util/netcdf_io/netcdf_io.cc

// header: util/netcdf_io/netcdf_io.h

#ifndef UTIL_NETCDF_IO_NETCDF_IO_H_
#define UTIL_NETCDF_IO_NETCDF_IO_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace netcdf_io {

absl::StatusOr<bool> NetcdfReadBool(int ncid, const std::string& variable_name);

absl::StatusOr<char> NetcdfReadChar(int ncid, const std::string& variable_name);

absl::StatusOr<int> NetcdfReadInt(int ncid, const std::string& variable_name);

absl::StatusOr<double> NetcdfReadDouble(int ncid,
const std::string& variable_name);

absl::StatusOr<std::string> NetcdfReadString(int ncid,
const std::string& variable_name);

absl::StatusOr<std::vector<double> > NetcdfReadArray1D(
int ncid, const std::string& variable_name);

absl::StatusOr<std::vector<std::vector<double> > > NetcdfReadArray2D(
int ncid, const std::string& variable_name);

absl::StatusOr<std::vector<std::vector<std::vector<double> > > >
NetcdfReadArray3D(int ncid, const std::string& variable_name);

}

#endif

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "netcdf.h"

namespace netcdf_io {

namespace {

absl::StatusOr<int> FindVariableId(int ncid, const std::string& variable_name) {
int variable_id = 0;
if (nc_inq_varid(ncid, variable_name.c_str(), &variable_id) != NC_NOERR) {
return absl::NotFoundError(
absl::StrFormat("variable '%s' not found", variable_name));
}
return variable_id;
}

absl::StatusOr<int> GetVariableRank(int ncid, int variable_id,
const std::string& variable_name) {
int rank = 0;
if (nc_inq_varndims(ncid, variable_id, &rank) != NC_NOERR) {
return absl::InternalError(absl::StrFormat(
"could not determine rank of variable '%s'", variable_name));
}
return rank;
}

absl::StatusOr<std::vector<size_t> > GetVariableDimensions(
int ncid, int variable_id, int rank, const std::string& variable_name) {
std::vector<int> dimension_ids(rank, 0);
if (nc_inq_vardimid(ncid, variable_id, dimension_ids.data()) != NC_NOERR) {
return absl::InternalError(absl::StrFormat(
"could not determine dimension ids of variable '%s'", variable_name));
}

std::vector<size_t> dimensions(rank, 0);
for (int i = 0; i < rank; ++i) {
size_t dimension = 0;
if (nc_inq_dimlen(ncid, dimension_ids[i], &dimension) != NC_NOERR) {
return absl::InternalError(
absl::StrFormat("could not determine dimension %d of variable '%s'",
i, variable_name));
}
dimensions[i] = dimension;
}
return dimensions;
}

}

absl::StatusOr<bool> NetcdfReadBool(int ncid,
const std::string& variable_name) {

const std::string logical_variable_name = variable_name + "__logical__";

absl::StatusOr<int> variable_id = FindVariableId(ncid, logical_variable_name);
if (!variable_id.ok()) {
return variable_id.status();
}

absl::StatusOr<int> rank =
GetVariableRank(ncid, *variable_id, logical_variable_name);
if (!rank.ok()) {
return rank.status();
}
if (*rank != 0) {
return absl::InvalidArgumentError(
absl::StrFormat("Not a rank-0 array: %s", logical_variable_name));
}

int variable_data = 0;
if (nc_get_var_int(ncid, *variable_id, &variable_data) != NC_NOERR) {
return absl::InternalError(
absl::StrFormat("could not read variable '%s'", logical_variable_name));
}

return variable_data != 0;
}

absl::StatusOr<char> NetcdfReadChar(int ncid,
const std::string& variable_name) {
absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
if (!variable_id.ok()) {
return variable_id.status();
}

absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
if (!rank.ok()) {
return rank.status();
}
if (*rank != 1) {
return absl::InvalidArgumentError(
absl::StrFormat("Not a rank-1 array: %s", variable_name));
}

absl::StatusOr<std::vector<size_t> > dimensions =
GetVariableDimensions(ncid, *variable_id, *rank, variable_name);
if (!dimensions.ok()) {
return dimensions.status();
}

if ((*dimensions)[0] != 1) {
return absl::InvalidArgumentError(
absl::StrFormat("Not a length-1 array: %s", variable_name));
}

std::vector<size_t> read_start_indices(*rank, 0);
std::vector<char> variable_data(1, 0);
if (nc_get_vara(ncid, *variable_id, read_start_indices.data(),
dimensions->data(), variable_data.data()) != NC_NOERR) {
return absl::InternalError(
absl::StrFormat("could not read variable '%s'", variable_name));
}

return variable_data[0];
}

absl::StatusOr<int> NetcdfReadInt(int ncid, const std::string& variable_name) {
absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
if (!variable_id.ok()) {
return variable_id.status();
}

absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
if (!rank.ok()) {
return rank.status();
}
if (*rank != 0) {
return absl::InvalidArgumentError(
absl::StrFormat("Not a rank-0 array: %s", variable_name));
}

int variable_data = 0;
if (nc_get_var_int(ncid, *variable_id, &variable_data) != NC_NOERR) {
return absl::InternalError(
absl::StrFormat("could not read variable '%s'", variable_name));
}

return variable_data;
}

absl::StatusOr<double> NetcdfReadDouble(int ncid,
const std::string& variable_name) {
absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
if (!variable_id.ok()) {
return variable_id.status();
}

absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
if (!rank.ok()) {
return rank.status();
}
if (*rank != 0) {
return absl::InvalidArgumentError(
absl::StrFormat("Not a rank-0 array: %s", variable_name));
}

double variable_data = 0;
if (nc_get_var_double(ncid, *variable_id, &variable_data) != NC_NOERR) {
return absl::InternalError(
absl::StrFormat("could not read variable '%s'", variable_name));
}

return variable_data;
}

absl::StatusOr<std::string> NetcdfReadString(int ncid,
const std::string& variable_name) {
absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
if (!variable_id.ok()) {
return variable_id.status();
}

absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
if (!rank.ok()) {
return rank.status();
}

if (*rank != 1) {
return absl::InvalidArgumentError(
absl::StrFormat("Not a rank-1 array: %s", variable_name));
}

absl::StatusOr<std::vector<size_t> > dimensions =
GetVariableDimensions(ncid, *variable_id, *rank, variable_name);
if (!dimensions.ok()) {
return dimensions.status();
}

size_t total_element_count = (*dimensions)[0];

std::vector<size_t> read_start_indices(*rank, 0);

std::vector<char> variable_data(total_element_count + 1, 0);
if (nc_get_vara(ncid, *variable_id, read_start_indices.data(),
dimensions->data(), variable_data.data()) != NC_NOERR) {
return absl::InternalError(
absl::StrFormat("could not read variable '%s'", variable_name));
}
std::string string_from_char_array = std::string(variable_data.data());

return std::string(absl::StripAsciiWhitespace(string_from_char_array));
}

absl::StatusOr<std::vector<double> > NetcdfReadArray1D(
int ncid, const std::string& variable_name) {
absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
if (!variable_id.ok()) {
return variable_id.status();
}

absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
if (!rank.ok()) {
return rank.status();
}
if (*rank != 1) {
return absl::InvalidArgumentError(
absl::StrFormat("Not a rank-1 array: %s", variable_name));
}

absl::StatusOr<std::vector<size_t> > dimensions =
GetVariableDimensions(ncid, *variable_id, *rank, variable_name);
if (!dimensions.ok()) {
return dimensions.status();
}

size_t total_element_count = (*dimensions)[0];

std::vector<size_t> read_start_indices(*rank, 0);
std::vector<double> variable_data(total_element_count, 0.0);
if (nc_get_vara(ncid, *variable_id, read_start_indices.data(),
dimensions->data(), variable_data.data()) != NC_NOERR) {
return absl::InternalError(
absl::StrFormat("could not read variable '%s'", variable_name));
}

return variable_data;
}

absl::StatusOr<std::vector<std::vector<double> > > NetcdfReadArray2D(
int ncid, const std::string& variable_name) {
absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
if (!variable_id.ok()) {
return variable_id.status();
}

absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
if (!rank.ok()) {
return rank.status();
}
if (*rank != 2) {
return absl::InvalidArgumentError(
absl::StrFormat("Not a rank-2 array: %s", variable_name));
}

absl::StatusOr<std::vector<size_t> > dimensions_or =
GetVariableDimensions(ncid, *variable_id, *rank, variable_name);
if (!dimensions_or.ok()) {
return dimensions_or.status();
}
auto dimensions = dimensions_or.value();

size_t total_element_count = dimensions[0] * dimensions[1];

std::vector<size_t> read_start_indices(*rank, 0);
std::vector<double> variable_data(total_element_count, 0.0);
if (nc_get_vara(ncid, *variable_id, read_start_indices.data(),
dimensions.data(), variable_data.data()) != NC_NOERR) {
return absl::InternalError(
absl::StrFormat("could not read variable '%s'", variable_name));
}

std::vector<std::vector<double> > two_dimensional_data(dimensions[0]);
for (size_t i = 0; i < dimensions[0]; ++i) {
two_dimensional_data[i].resize(dimensions[1], 0.0);
for (size_t j = 0; j < dimensions[1]; ++j) {
two_dimensional_data[i][j] = variable_data[i * dimensions[1] + j];
}
}

return two_dimensional_data;
}

absl::StatusOr<std::vector<std::vector<std::vector<double> > > >
NetcdfReadArray3D(int ncid, const std::string& variable_name) {
absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
if (!variable_id.ok()) {
return variable_id.status();
}

absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
if (!rank.ok()) {
return rank.status();
}
if (*rank != 3) {
return absl::InvalidArgumentError(
absl::StrFormat("Not a rank-3 array: %s", variable_name));
}

absl::StatusOr<std::vector<size_t> > dimensions_or =
GetVariableDimensions(ncid, *variable_id, *rank, variable_name);
if (!dimensions_or.ok()) {
return dimensions_or.status();
}
auto dimensions = dimensions_or.value();

size_t total_element_count = dimensions[0] * dimensions[1] * dimensions[2];
std::vector<size_t> read_start_indices(*rank, 0);
std::vector<double> variable_data(total_element_count, 0.0);
if (nc_get_vara(ncid, *variable_id, read_start_indices.data(),
dimensions.data(), variable_data.data()) != NC_NOERR) {
return absl::InternalError(
absl::StrFormat("could not read variable '%s'", variable_name));
}

std::vector<std::vector<std::vector<double> > > three_dimensional_data(
dimensions[0]);
for (size_t i = 0; i < dimensions[0]; ++i) {
three_dimensional_data[i].resize(dimensions[1]);
for (size_t j = 0; j < dimensions[1]; ++j) {
three_dimensional_data[i][j].resize(dimensions[2]);
for (size_t k = 0; k < dimensions[2]; ++k) {
three_dimensional_data[i][j][k] =
variable_data[(i * dimensions[1] + j) * dimensions[2] + k];
}
}
}

return three_dimensional_data;
}

}

// source: util/testing/numerical_comparison_lib.cc

// header: util/testing/numerical_comparison_lib.h

#ifndef UTIL_TESTING_NUMERICAL_COMPARISON_LIB_H_
#define UTIL_TESTING_NUMERICAL_COMPARISON_LIB_H_

#include <Eigen/Dense>
#include <vector>

namespace testing {

bool IsCloseRelAbs(double expected, double actual, double tolerance);

bool IsVectorCloseRelAbs(const std::vector<double>& expected,
const std::vector<double>& actual, double tolerance);

bool IsVectorCloseRelAbs(const Eigen::VectorXd& expected,
const Eigen::VectorXd& actual, double tolerance);
}

#endif

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

#include "absl/strings/str_format.h"

bool testing::IsCloseRelAbs(double expected, double actual, double tolerance) {
const double rel_abs_error = (actual - expected) / (1.0 + std::abs(expected));
if (std::abs(rel_abs_error) > tolerance ||
(std::isnan(expected) != std::isnan(actual))) {
std::cerr << absl::StrFormat(
"out-of-tolerance: |% .3e| > % .3e\n  expected = % .20e\n    "
"actual = % .20e\n",
rel_abs_error, tolerance, expected, actual);
return false;
}
return true;
}

bool testing::IsVectorCloseRelAbs(const std::vector<double>& expected,
const std::vector<double>& actual,
double tolerance) {
const auto& expected_eigen = Eigen::Map<const Eigen::VectorXd>(
expected.data(), static_cast<Eigen::Index>(expected.size()));
const auto& actual_eigen = Eigen::Map<const Eigen::VectorXd>(
actual.data(), static_cast<Eigen::Index>(actual.size()));

return IsVectorCloseRelAbs(expected_eigen, actual_eigen, tolerance);
}

bool testing::IsVectorCloseRelAbs(const Eigen::VectorXd& expected,
const Eigen::VectorXd& actual,
double tolerance) {
if (expected.size() != actual.size()) {
return false;
}

for (int i = 0; i < expected.size(); ++i) {
if (!IsCloseRelAbs(expected[i], actual[i], tolerance)) {
return false;
}
}

return true;
}

// source: vmecpp/common/composed_types_lib/composed_types_lib.cc

// header: vmecpp/common/composed_types_lib/composed_types_lib.h

#ifndef VMECPP_COMMON_COMPOSED_TYPES_LIB_COMPOSED_TYPES_LIB_H_
#define VMECPP_COMMON_COMPOSED_TYPES_LIB_COMPOSED_TYPES_LIB_H_

#include <filesystem>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

// header: vmecpp/common/composed_types_definition/composed_types.h

#ifndef VMECPP_COMMON_COMPOSED_TYPES_DEFINITION_COMPOSED_TYPES_H_
#define VMECPP_COMMON_COMPOSED_TYPES_DEFINITION_COMPOSED_TYPES_H_

#include <list>

namespace composed_types {

struct Vector3d {

bool has_x_ = false;
double x_ = 0.0;

bool has_y_ = false;
double y_ = 0.0;

bool has_z_ = false;
double z_ = 0.0;

bool has_x() const { return has_x_; }
double x() const { return x_; }
void set_x(double value) {
x_ = value;
has_x_ = true;
}
void clear_x() {
x_ = 0.0;
has_x_ = false;
}

bool has_y() const { return has_y_; }
double y() const { return y_; }
void set_y(double value) {
y_ = value;
has_y_ = true;
}
void clear_y() {
y_ = 0.0;
has_y_ = false;
}

bool has_z() const { return has_z_; }
double z() const { return z_; }
void set_z(double value) {
z_ = value;
has_z_ = true;
}
void clear_z() {
z_ = 0.0;
has_z_ = false;
}

void Clear() {
clear_x();
clear_y();
clear_z();
}

void CopyFrom(const Vector3d& other) {
has_x_ = other.has_x_;
x_ = other.x_;

has_y_ = other.has_y_;
y_ = other.y_;

has_z_ = other.has_z_;
z_ = other.z_;
}
};

struct FourierCoefficient1D {

bool has_fc_cos_ = false;
double fc_cos_ = 0.0;

bool has_fc_sin_ = false;
double fc_sin_ = 0.0;

bool has_mode_number_ = false;
int mode_number_ = 0;

bool has_fc_cos() const { return has_fc_cos_; }
double fc_cos() const { return fc_cos_; }
void set_fc_cos(double value) {
fc_cos_ = value;
has_fc_cos_ = true;
}
void clear_fc_cos() {
fc_cos_ = 0.0;
has_fc_cos_ = false;
}

bool has_fc_sin() const { return has_fc_sin_; }
double fc_sin() const { return fc_sin_; }
void set_fc_sin(double value) {
fc_sin_ = value;
has_fc_sin_ = true;
}
void clear_fc_sin() {
fc_sin_ = 0.0;
has_fc_sin_ = false;
}

bool has_mode_number() const { return has_mode_number_; }
int mode_number() const { return mode_number_; }
void set_mode_number(int value) {
mode_number_ = value;
has_mode_number_ = true;
}
void clear_mode_number() {
mode_number_ = 0;
has_mode_number_ = false;
}

void Clear() {
clear_fc_cos();
clear_fc_sin();
clear_mode_number();
}
};

struct FourierCoefficient2D {

bool has_fc_cos_ = false;
double fc_cos_ = 0.0;

bool has_fc_sin_ = false;
double fc_sin_ = 0.0;

bool has_poloidal_mode_number_ = false;
int poloidal_mode_number_ = 0;

bool has_toroidal_mode_number_ = false;
int toroidal_mode_number_ = 0;

bool has_fc_cos() const { return has_fc_cos_; }
double fc_cos() const { return fc_cos_; }
void set_fc_cos(double value) {
fc_cos_ = value;
has_fc_cos_ = true;
}
void clear_fc_cos() {
fc_cos_ = 0.0;
has_fc_cos_ = false;
}

bool has_fc_sin() const { return has_fc_sin_; }
double fc_sin() const { return fc_sin_; }
void set_fc_sin(double value) {
fc_sin_ = value;
has_fc_sin_ = true;
}
void clear_fc_sin() {
fc_sin_ = 0.0;
has_fc_sin_ = false;
}

bool has_poloidal_mode_number() const { return has_poloidal_mode_number_; }
int poloidal_mode_number() const { return poloidal_mode_number_; }
void set_poloidal_mode_number(int value) {
poloidal_mode_number_ = value;
has_poloidal_mode_number_ = true;
}
void clear_poloidal_mode_number() {
poloidal_mode_number_ = 0;
has_poloidal_mode_number_ = false;
}

bool has_toroidal_mode_number() const { return has_toroidal_mode_number_; }
int toroidal_mode_number() const { return toroidal_mode_number_; }
void set_toroidal_mode_number(int value) {
toroidal_mode_number_ = value;
has_toroidal_mode_number_ = true;
}
void clear_toroidal_mode_number() {
toroidal_mode_number_ = 0;
has_toroidal_mode_number_ = false;
}

void Clear() {
clear_fc_cos();
clear_fc_sin();
clear_poloidal_mode_number();
clear_toroidal_mode_number();
}
};

struct CurveRZFourier {

std::list<composed_types::FourierCoefficient1D> r_;

std::list<composed_types::FourierCoefficient1D> z_;

int r_size() const { return static_cast<int>(r_.size()); }
const composed_types::FourierCoefficient1D& r(int index) const {
auto it = r_.cbegin();
std::advance(it, index);
return *it;
}
composed_types::FourierCoefficient1D* mutable_r(int index) {
auto it = r_.begin();
std::advance(it, index);
return &(*it);
}
composed_types::FourierCoefficient1D* add_r() {
r_.emplace_back();
auto it = r_.end();
--it;
return &(*it);
}
const std::list<composed_types::FourierCoefficient1D>& r() const {
return r_;
}
std::list<composed_types::FourierCoefficient1D>* mutable_r() { return &r_; }
void clear_r() { r_.clear(); }

int z_size() const { return static_cast<int>(z_.size()); }
const composed_types::FourierCoefficient1D& z(int index) const {
auto it = z_.cbegin();
std::advance(it, index);
return *it;
}
composed_types::FourierCoefficient1D* mutable_z(int index) {
auto it = z_.begin();
std::advance(it, index);
return &(*it);
}
composed_types::FourierCoefficient1D* add_z() {
z_.emplace_back();
auto it = z_.end();
--it;
return &(*it);
}
const std::list<composed_types::FourierCoefficient1D>& z() const {
return z_;
}
std::list<composed_types::FourierCoefficient1D>* mutable_z() { return &z_; }
void clear_z() { z_.clear(); }

void Clear() {
clear_r();
clear_z();
}
};

struct SurfaceRZFourier {

std::list<composed_types::FourierCoefficient2D> r_;

std::list<composed_types::FourierCoefficient2D> z_;

int r_size() const { return static_cast<int>(r_.size()); }
const composed_types::FourierCoefficient2D& r(int index) const {
auto it = r_.cbegin();
std::advance(it, index);
return *it;
}
composed_types::FourierCoefficient2D* mutable_r(int index) {
auto it = r_.begin();
std::advance(it, index);
return &(*it);
}
composed_types::FourierCoefficient2D* add_r() {
r_.emplace_back();
auto it = r_.end();
--it;
return &(*it);
}
const std::list<composed_types::FourierCoefficient2D>& r() const {
return r_;
}
std::list<composed_types::FourierCoefficient2D>* mutable_r() { return &r_; }
void clear_r() { r_.clear(); }

int z_size() const { return static_cast<int>(z_.size()); }
const composed_types::FourierCoefficient2D& z(int index) const {
auto it = z_.cbegin();
std::advance(it, index);
return *it;
}
composed_types::FourierCoefficient2D* mutable_z(int index) {
auto it = z_.begin();
std::advance(it, index);
return &(*it);
}
composed_types::FourierCoefficient2D* add_z() {
z_.emplace_back();
auto it = z_.end();
--it;
return &(*it);
}
const std::list<composed_types::FourierCoefficient2D>& z() const {
return z_;
}
std::list<composed_types::FourierCoefficient2D>* mutable_z() { return &z_; }
void clear_z() { z_.clear(); }

void Clear() {
clear_r();
clear_z();
}
};

}

#endif

namespace composed_types {

absl::Status IsVector3dFullyPopulated(const Vector3d& vector,
absl::string_view vector_name = "");

double Length(const Vector3d& vector);

Vector3d ScaleTo(const Vector3d& vector, double desired_length);

Vector3d Normalize(const Vector3d& vector);

Vector3d Add(const Vector3d& vector_1, const Vector3d& vector_2);

Vector3d Subtract(const Vector3d& vector_1, const Vector3d& vector_2);

double DotProduct(const Vector3d& vector_1, const Vector3d& vector_2);

Vector3d CrossProduct(const Vector3d& vector_1, const Vector3d& vector_2);

Vector3d MostPerpendicularCoordinateAxis(const Vector3d& axis);

std::array<Vector3d, 3> OrthonormalFrameAroundAxis(const Vector3d& axis);

absl::Status IsFourierCoefficient1DFullyPopulated(
const FourierCoefficient1D& fourier_coefficient,
absl::string_view fourier_coefficient_name = "");

absl::Status IsFourierCoefficient2DFullyPopulated(
const FourierCoefficient2D& fourier_coefficient,
absl::string_view fourier_coefficient_name = "");

absl::Status IsCurveRZFourierFullyPopulated(const CurveRZFourier& curve);
absl::StatusOr<CurveRZFourier> CurveRZFourierFromCsv(
const std::string& axis_coefficients_csv);
absl::StatusOr<std::string> CurveRZFourierToCsv(const CurveRZFourier& axis);
absl::StatusOr<std::vector<int>> ModeNumbers(const CurveRZFourier& curve);
absl::StatusOr<std::vector<double>> CoefficientsRCos(
const CurveRZFourier& curve);
absl::StatusOr<std::vector<double>> CoefficientsZSin(
const CurveRZFourier& curve);
absl::StatusOr<std::vector<double>> CoefficientsRSin(
const CurveRZFourier& curve);
absl::StatusOr<std::vector<double>> CoefficientsZCos(
const CurveRZFourier& curve);

absl::Status IsSurfaceRZFourierFullyPopulated(const SurfaceRZFourier& surface);
absl::StatusOr<SurfaceRZFourier> SurfaceRZFourierFromCsv(
const std::string& boundary_coefficients_csv);
absl::StatusOr<std::string> SurfaceRZFourierToCsv(
const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<int>> PoloidalModeNumbers(
const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<int>> ToroidalModeNumbers(
const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<double>> CoefficientsRCos(
const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<double>> CoefficientsZSin(
const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<double>> CoefficientsRSin(
const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<double>> CoefficientsZCos(
const SurfaceRZFourier& surface);

}

#endif

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"

namespace composed_types {

absl::Status IsVector3dFullyPopulated(const Vector3d& vector,
absl::string_view vector_name) {
if (!vector.has_x()) {
std::stringstream error_message;
error_message << vector_name;
error_message << " has no x component.";
return absl::NotFoundError(error_message.str());
}

if (!vector.has_y()) {
std::stringstream error_message;
error_message << vector_name;
error_message << " has no y component.";
return absl::NotFoundError(error_message.str());
}

if (!vector.has_z()) {
std::stringstream error_message;
error_message << vector_name;
error_message << " has no z component.";
return absl::NotFoundError(error_message.str());
}

return absl::OkStatus();
}

double Length(const Vector3d& vector) {
return std::hypot(vector.x(), vector.y(), vector.z());
}

Vector3d ScaleTo(const Vector3d& vector, double desired_length) {
const double vector_length = Length(vector);
const double scaling_factor = desired_length / vector_length;
Vector3d normalized_vector;
normalized_vector.set_x(vector.x() * scaling_factor);
normalized_vector.set_y(vector.y() * scaling_factor);
normalized_vector.set_z(vector.z() * scaling_factor);
return normalized_vector;
}

Vector3d Normalize(const Vector3d& vector) {
return ScaleTo(vector, 1.0);
}

Vector3d Add(const Vector3d& vector_1, const Vector3d& vector_2) {
Vector3d sum;
sum.set_x(vector_1.x() + vector_2.x());
sum.set_y(vector_1.y() + vector_2.y());
sum.set_z(vector_1.z() + vector_2.z());
return sum;
}

Vector3d Subtract(const Vector3d& vector_1, const Vector3d& vector_2) {
Vector3d difference;
difference.set_x(vector_1.x() - vector_2.x());
difference.set_y(vector_1.y() - vector_2.y());
difference.set_z(vector_1.z() - vector_2.z());
return difference;
}

double DotProduct(const Vector3d& vector_1, const Vector3d& vector_2) {
double dot_product = 0.0;
dot_product += vector_1.x() * vector_2.x();
dot_product += vector_1.y() * vector_2.y();
dot_product += vector_1.z() * vector_2.z();
return dot_product;
}

Vector3d CrossProduct(const Vector3d& vector_1, const Vector3d& vector_2) {
Vector3d cross_product;
cross_product.set_x(vector_1.y() * vector_2.z() -
vector_1.z() * vector_2.y());
cross_product.set_y(vector_1.z() * vector_2.x() -
vector_1.x() * vector_2.z());
cross_product.set_z(vector_1.x() * vector_2.y() -
vector_1.y() * vector_2.x());
return cross_product;
}

Vector3d MostPerpendicularCoordinateAxis(const Vector3d& axis) {

const double min_component = std::min(
std::min(std::abs(axis.x()), std::abs(axis.y())), std::abs(axis.z()));

Vector3d most_perpendicular_axis;
if (min_component == std::abs(axis.x())) {
most_perpendicular_axis.set_x(1.0);
} else if (min_component == std::abs(axis.y())) {
most_perpendicular_axis.set_y(1.0);
} else {
most_perpendicular_axis.set_z(1.0);
}

return most_perpendicular_axis;
}

std::array<Vector3d, 3> OrthonormalFrameAroundAxis(const Vector3d& axis) {
std::array<Vector3d, 3> orthonormal_frame;

orthonormal_frame[0] = ScaleTo(axis, 1.0);

const Vector3d most_perpendicular_axis =
MostPerpendicularCoordinateAxis(axis);
const double axis_dot_most_perp =
DotProduct(orthonormal_frame[0], most_perpendicular_axis);
orthonormal_frame[1] =
Add(most_perpendicular_axis,
ScaleTo(orthonormal_frame[0], -axis_dot_most_perp));

orthonormal_frame[2] =
CrossProduct(orthonormal_frame[0], orthonormal_frame[1]);

return orthonormal_frame;
}

absl::Status IsFourierCoefficient1DFullyPopulated(
const FourierCoefficient1D& fourier_coefficient,
absl::string_view fourier_coefficient_name) {
if (!fourier_coefficient.has_fc_cos() && !fourier_coefficient.has_fc_sin()) {
std::stringstream error_message;
error_message << fourier_coefficient_name;
error_message << " has neither a cosine nor a sine coefficient.";
return absl::NotFoundError(error_message.str());
}

if (!fourier_coefficient.has_mode_number()) {
std::stringstream error_message;
error_message << fourier_coefficient_name;
error_message << " has no mode number set.";
return absl::NotFoundError(error_message.str());
}

return absl::OkStatus();
}

absl::Status IsFourierCoefficient2DFullyPopulated(
const FourierCoefficient2D& fourier_coefficient,
absl::string_view fourier_coefficient_name) {
if (!fourier_coefficient.has_fc_cos() && !fourier_coefficient.has_fc_sin()) {
std::stringstream error_message;
error_message << fourier_coefficient_name;
error_message << " has neither a cosine nor a sine coefficient.";
return absl::NotFoundError(error_message.str());
}

if (!fourier_coefficient.has_poloidal_mode_number()) {
std::stringstream error_message;
error_message << fourier_coefficient_name;
error_message << " has no poloidal mode number set.";
return absl::NotFoundError(error_message.str());
}

if (!fourier_coefficient.has_toroidal_mode_number()) {
std::stringstream error_message;
error_message << fourier_coefficient_name;
error_message << " has no toroidal mode number set.";
return absl::NotFoundError(error_message.str());
}

return absl::OkStatus();
}

absl::Status IsCurveRZFourierFullyPopulated(const CurveRZFourier& curve) {
if (curve.r_size() == 0) {
return absl::NotFoundError(
"CurveRZFourier has no FourierCoefficient1D for R");
}

if (curve.z_size() == 0) {
return absl::NotFoundError(
"CurveRZFourier has no FourierCoefficient1D for Z");
}

if (curve.r_size() != curve.z_size()) {
std::stringstream error_message;
error_message << "CurveRZFourier has different number of Fourier "
"coefficients for R (";
error_message << curve.r_size();
error_message << ") and z(";
error_message << curve.z_size();
error_message << ")";
return absl::NotFoundError(error_message.str());
}

const int rz_size = curve.r_size();
for (int i = 0; i < rz_size; ++i) {
absl::Status r_status = IsFourierCoefficient1DFullyPopulated(
curve.r(i), absl::StrFormat("r[%d]", i));
if (!r_status.ok()) {
return r_status;
}

absl::Status z_status = IsFourierCoefficient1DFullyPopulated(
curve.z(i), absl::StrFormat("z[%d]", i));
if (!z_status.ok()) {
return z_status;
}

if (curve.r(i).mode_number() != curve.z(i).mode_number()) {
return absl::NotFoundError(absl::StrFormat(
"found different mode numbers at coefficient %d for r(%d) and z(%d)",
i, curve.r(i).mode_number(), curve.z(i).mode_number()));
}
}

return absl::OkStatus();
}

absl::StatusOr<CurveRZFourier> CurveRZFourierFromCsv(
const std::string& axis_coefficients_csv) {
CurveRZFourier axis_coefficients;

std::stringstream axis_coefficients_ss(axis_coefficients_csv);

std::string header_line;
if (!std::getline(axis_coefficients_ss, header_line)) {
return absl::InvalidArgumentError("cannot read header line");
}

if (absl::StripAsciiWhitespace(header_line) !=
"n,raxis_c,zaxis_s,raxis_s,zaxis_c") {
return absl::NotFoundError(
"header line 'n,raxis_c,zaxis_s,raxis_s,zaxis_c' not found");
}

for (std::string raw_line; std::getline(axis_coefficients_ss, raw_line);
) {
absl::string_view stripped_line = absl::StripAsciiWhitespace(raw_line);
std::vector<std::string> line_parts = absl::StrSplit(
stripped_line, absl::ByAnyChar(","), absl::SkipWhitespace());
if (line_parts.size() == 5) {
const int n = std::stoi(line_parts[0]);

FourierCoefficient1D* raxis = axis_coefficients.add_r();
raxis->set_mode_number(n);

FourierCoefficient1D* zaxis = axis_coefficients.add_z();
zaxis->set_mode_number(n);

raxis->set_fc_cos(std::stod(line_parts[1]));
zaxis->set_fc_sin(std::stod(line_parts[2]));
raxis->set_fc_sin(std::stod(line_parts[3]));
zaxis->set_fc_cos(std::stod(line_parts[4]));
} else {
std::stringstream error_message;
error_message
<< "cannot parse line: '" << stripped_line << "': has "
<< line_parts.size()
<< " parts, but expect 5: n, raxis_c, zaxis_s, raxis_s, zaxis_c";
return absl::InvalidArgumentError(error_message.str());
}
}

return axis_coefficients;
}

absl::StatusOr<std::string> CurveRZFourierToCsv(const CurveRZFourier& axis) {

absl::Status status = IsCurveRZFourierFullyPopulated(axis);
if (!status.ok()) {
return status;
}

std::stringstream ss;

ss << "n,raxis_c,zaxis_s,raxis_s,zaxis_c\n";

const int rz_size = axis.r_size();
for (int i = 0; i < rz_size; ++i) {
FourierCoefficient1D r = axis.r(i);
FourierCoefficient1D z = axis.z(i);
const int n = r.mode_number();
ss << n << ",";
ss << std::setprecision(16) << r.fc_cos() << ",";
ss << std::setprecision(16) << z.fc_sin() << ",";
ss << std::setprecision(16) << r.fc_sin() << ",";
ss << std::setprecision(16) << z.fc_cos() << "\n";
}

return ss.str();
}

absl::StatusOr<std::vector<int>> ModeNumbers(const CurveRZFourier& curve) {
absl::Status is_fully_populated = IsCurveRZFourierFullyPopulated(curve);
if (!is_fully_populated.ok()) {
return is_fully_populated;
}
int rz_size = curve.r_size();
std::vector<int> mode_numbers(rz_size);
for (int i = 0; i < rz_size; ++i) {
mode_numbers[i] = curve.r(i).mode_number();
}
return mode_numbers;
}

absl::StatusOr<std::vector<double>> CoefficientsRCos(
const CurveRZFourier& curve) {
absl::Status is_fully_populated = IsCurveRZFourierFullyPopulated(curve);
if (!is_fully_populated.ok()) {
return is_fully_populated;
}
int rz_size = curve.r_size();
std::vector<double> r_cos(rz_size);
for (int i = 0; i < rz_size; ++i) {
r_cos[i] = curve.r(i).fc_cos();
}
return r_cos;
}

absl::StatusOr<std::vector<double>> CoefficientsZSin(
const CurveRZFourier& curve) {
absl::Status is_fully_populated = IsCurveRZFourierFullyPopulated(curve);
if (!is_fully_populated.ok()) {
return is_fully_populated;
}
int rz_size = curve.r_size();
std::vector<double> z_sin(rz_size);
for (int i = 0; i < rz_size; ++i) {
z_sin[i] = curve.z(i).fc_sin();
}
return z_sin;
}

absl::StatusOr<std::vector<double>> CoefficientsRSin(
const CurveRZFourier& curve) {
absl::Status is_fully_populated = IsCurveRZFourierFullyPopulated(curve);
if (!is_fully_populated.ok()) {
return is_fully_populated;
}
int rz_size = curve.r_size();
std::vector<double> r_sin(rz_size);
for (int i = 0; i < rz_size; ++i) {
r_sin[i] = curve.r(i).fc_sin();
}
return r_sin;
}

absl::StatusOr<std::vector<double>> CoefficientsZCos(
const CurveRZFourier& curve) {
absl::Status is_fully_populated = IsCurveRZFourierFullyPopulated(curve);
if (!is_fully_populated.ok()) {
return is_fully_populated;
}
int rz_size = curve.r_size();
std::vector<double> z_cos(rz_size);
for (int i = 0; i < rz_size; ++i) {
z_cos[i] = curve.z(i).fc_cos();
}
return z_cos;
}

absl::Status IsSurfaceRZFourierFullyPopulated(const SurfaceRZFourier& surface) {
if (surface.r_size() == 0) {
return absl::NotFoundError(
"SurfaceRZFourier has no FourierCoefficient1D for R");
}

if (surface.z_size() == 0) {
return absl::NotFoundError(
"SurfaceRZFourier has no FourierCoefficient1D for Z");
}

if (surface.r_size() != surface.z_size()) {
std::stringstream error_message;
error_message << "SurfaceRZFourier has different number of Fourier "
"coefficients for R (";
error_message << surface.r_size();
error_message << ") and z(";
error_message << surface.z_size();
error_message << ")";
return absl::NotFoundError(error_message.str());
}

const int rz_size = surface.r_size();
for (int i = 0; i < rz_size; ++i) {
absl::Status r_status = IsFourierCoefficient2DFullyPopulated(
surface.r(i), absl::StrFormat("r[%d]", i));
if (!r_status.ok()) {
return r_status;
}

absl::Status z_status = IsFourierCoefficient2DFullyPopulated(
surface.z(i), absl::StrFormat("z[%d]", i));
if (!z_status.ok()) {
return z_status;
}

if (surface.r(i).poloidal_mode_number() !=
surface.z(i).poloidal_mode_number()) {
return absl::NotFoundError(
absl::StrFormat("found different poloidal mode numbers at "
"coefficient %d for r(%d) and z(%d)",
i, surface.r(i).poloidal_mode_number(),
surface.z(i).poloidal_mode_number()));
}

if (surface.r(i).toroidal_mode_number() !=
surface.z(i).toroidal_mode_number()) {
return absl::NotFoundError(
absl::StrFormat("found different toroidal mode numbers at "
"coefficient %d for r(%d) and z(%d)",
i, surface.r(i).toroidal_mode_number(),
surface.z(i).toroidal_mode_number()));
}
}

return absl::OkStatus();
}

absl::StatusOr<SurfaceRZFourier> SurfaceRZFourierFromCsv(
const std::string& boundary_coefficients_csv) {
SurfaceRZFourier boundary_coefficients;

std::stringstream boundary_coefficients_ss(boundary_coefficients_csv);

std::string header_line;
if (!std::getline(boundary_coefficients_ss, header_line)) {
return absl::InvalidArgumentError("cannot read header line");
}

if (absl::StripAsciiWhitespace(header_line) != "n,m,rbc,zbs,rbs,zbc") {
return absl::NotFoundError("header line 'n,m,rbc,zbs,rbs,zbc' not found");
}

for (std::string raw_line; std::getline(boundary_coefficients_ss, raw_line);
) {
absl::string_view stripped_line = absl::StripAsciiWhitespace(raw_line);
std::vector<std::string> line_parts = absl::StrSplit(
stripped_line, absl::ByAnyChar(","), absl::SkipWhitespace());
if (line_parts.size() == 6) {
const int n = std::stoi(line_parts[0]);
const int m = std::stoi(line_parts[1]);

FourierCoefficient2D* rbc_rbs = boundary_coefficients.add_r();
rbc_rbs->set_poloidal_mode_number(m);
rbc_rbs->set_toroidal_mode_number(n);

FourierCoefficient2D* zbs_zbc = boundary_coefficients.add_z();
zbs_zbc->set_poloidal_mode_number(m);
zbs_zbc->set_toroidal_mode_number(n);

rbc_rbs->set_fc_cos(std::stod(line_parts[2]));
zbs_zbc->set_fc_sin(std::stod(line_parts[3]));
rbc_rbs->set_fc_sin(std::stod(line_parts[4]));
zbs_zbc->set_fc_cos(std::stod(line_parts[5]));
} else {
std::stringstream error_message;
error_message << "cannot parse line: '" << stripped_line << "': has "
<< line_parts.size()
<< " parts, but expect 6: n, m, rbc, zbs, rbs, zbc";
return absl::InvalidArgumentError(error_message.str());
}
}

return boundary_coefficients;
}

absl::StatusOr<std::string> SurfaceRZFourierToCsv(
const SurfaceRZFourier& surface) {

absl::Status status = IsSurfaceRZFourierFullyPopulated(surface);
if (!status.ok()) {
return status;
}

std::stringstream ss;

ss << "n,m,rbc,zbs,rbs,zbc\n";

const int rz_size = surface.r_size();
for (int i = 0; i < rz_size; ++i) {
const FourierCoefficient2D& r = surface.r(i);
const FourierCoefficient2D& z = surface.z(i);

const int m = r.poloidal_mode_number();
const int n = r.toroidal_mode_number();

ss << n << "," << m << ",";
ss << std::setprecision(16) << r.fc_cos() << ",";
ss << std::setprecision(16) << z.fc_sin() << ",";
ss << std::setprecision(16) << r.fc_sin() << ",";
ss << std::setprecision(16) << z.fc_cos() << "\n";
}

return ss.str();
}

absl::StatusOr<std::vector<int>> PoloidalModeNumbers(
const SurfaceRZFourier& surface) {
absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
if (!is_fully_populated.ok()) {
return is_fully_populated;
}
int rz_size = surface.r_size();
std::vector<int> poloidal_mode_numbers(rz_size);
for (int i = 0; i < rz_size; ++i) {
poloidal_mode_numbers[i] = surface.r(i).poloidal_mode_number();
}
return poloidal_mode_numbers;
}

absl::StatusOr<std::vector<int>> ToroidalModeNumbers(
const SurfaceRZFourier& surface) {
absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
if (!is_fully_populated.ok()) {
return is_fully_populated;
}
int rz_size = surface.r_size();
std::vector<int> toroidal_mode_numbers(rz_size);
for (int i = 0; i < rz_size; ++i) {
toroidal_mode_numbers[i] = surface.r(i).toroidal_mode_number();
}
return toroidal_mode_numbers;
}

absl::StatusOr<std::vector<double>> CoefficientsRCos(
const SurfaceRZFourier& surface) {
absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
if (!is_fully_populated.ok()) {
return is_fully_populated;
}
int rz_size = surface.r_size();
std::vector<double> r_cos(rz_size);
for (int i = 0; i < rz_size; ++i) {
r_cos[i] = surface.r(i).fc_cos();
}
return r_cos;
}

absl::StatusOr<std::vector<double>> CoefficientsZSin(
const SurfaceRZFourier& surface) {
absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
if (!is_fully_populated.ok()) {
return is_fully_populated;
}
int rz_size = surface.r_size();
std::vector<double> z_sin(rz_size);
for (int i = 0; i < rz_size; ++i) {
z_sin[i] = surface.z(i).fc_sin();
}
return z_sin;
}

absl::StatusOr<std::vector<double>> CoefficientsRSin(
const SurfaceRZFourier& surface) {
absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
if (!is_fully_populated.ok()) {
return is_fully_populated;
}
int rz_size = surface.r_size();
std::vector<double> r_sin(rz_size);
for (int i = 0; i < rz_size; ++i) {
r_sin[i] = surface.r(i).fc_sin();
}
return r_sin;
}

absl::StatusOr<std::vector<double>> CoefficientsZCos(
const SurfaceRZFourier& surface) {
absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
if (!is_fully_populated.ok()) {
return is_fully_populated;
}
int rz_size = surface.r_size();
std::vector<double> z_cos(rz_size);
for (int i = 0; i < rz_size; ++i) {
z_cos[i] = surface.z(i).fc_cos();
}
return z_cos;
}

}

// source: vmecpp/common/flow_control/flow_control.cc

// header: vmecpp/common/flow_control/flow_control.h

#ifndef VMECPP_COMMON_FLOW_CONTROL_FLOW_CONTROL_H_
#define VMECPP_COMMON_FLOW_CONTROL_FLOW_CONTROL_H_

#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <vector>

namespace vmecpp {

enum class RestartReason : std::uint8_t {

NO_RESTART = 1,

BAD_JACOBIAN = 2,

BAD_PROGRESS = 3,

HUGE_INITIAL_FORCES = 4
};

RestartReason RestartReasonFromInt(int restart_reason);

class FlowControl {
public:

static constexpr int kPreconditionerUpdateInterval = 25;

FlowControl(bool lfreeb, double delt, int num_grids,
std::optional<int> max_threads = std::nullopt);

int max_threads() const;

const bool lfreeb;

RestartReason restart_reason;

int ns;

int neqs;
int neqs_old;

bool haveToFlipTheta;

int ijacob;

int multi_ns_grid;

int nsval;

double deltaS;

double ftolv;

int niterv;

int num_surfaces_to_distribute;

int ns_min;
int ns_old;

double delt0r;

double fsqr, fsqz, fsql;

std::vector<double> force_residual_r;
std::vector<double> force_residual_z;
std::vector<double> force_residual_lambda;

double fsqr1, fsqz1, fsql1;
double fsq;

std::vector<double> mhd_energy;

std::vector<double> delbsq;

std::vector<RestartReason> restart_reasons;

double res0;

double res1;

Eigen::Vector3d fResInvar;
Eigen::Vector3d fResPrecd;

private:
const int max_threads_;
};

}

#endif

#include "absl/log/check.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace vmecpp {

RestartReason RestartReasonFromInt(int restart_reason) {
switch (restart_reason) {
case 1:
return RestartReason::NO_RESTART;
case 2:
return RestartReason::BAD_JACOBIAN;
case 3:
return RestartReason::BAD_PROGRESS;
case 4:
return RestartReason::HUGE_INITIAL_FORCES;
default:
__builtin_unreachable();
}
}

int get_max_threads(std::optional<int> max_threads) {
if (max_threads == std::nullopt) {
#ifdef _OPENMP
return omp_get_max_threads();
#endif

return 1;
}
CHECK_GT(max_threads.value(), 0)
<< "The number of threads must be >=1. "
"To automatically use all available threads, pass std::nullopt";
return max_threads.value();
}

FlowControl::FlowControl(bool lfreeb, double delt, int num_grids,
std::optional<int> max_threads)
: lfreeb(lfreeb), max_threads_(get_max_threads(max_threads)) {
fsq = 1.0;

fsqr = 1.0;
fsqz = 1.0;
ftolv = fsqr;
ijacob = 0;
restart_reason = RestartReason::NO_RESTART;
res0 = -1;
res1 = -1;
delt0r = delt;
multi_ns_grid = num_grids;
neqs_old = 0;

fResInvar.setZero();
fResPrecd.setZero();

ns_old = 0;
}

int FlowControl::max_threads() const { return max_threads_; }

}

// source: vmecpp/common/fourier_basis/fourier_basis.cc

// header: vmecpp/common/fourier_basis/fourier_basis.h

#ifndef VMECPP_COMMON_FOURIER_BASIS_FOURIER_BASIS_H_
#define VMECPP_COMMON_FOURIER_BASIS_FOURIER_BASIS_H_

#include <Eigen/Dense>
#include <span>

// header: vmecpp/common/sizes/sizes.h

#ifndef VMECPP_COMMON_SIZES_SIZES_H_
#define VMECPP_COMMON_SIZES_SIZES_H_

#include <Eigen/Dense>

// header: vmecpp/common/util/util.h

#ifndef VMECPP_COMMON_UTIL_UTIL_H_
#define VMECPP_COMMON_UTIL_UTIL_H_

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "absl/log/check.h"

// header: vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h

#ifndef VMECPP_VMEC_VMEC_CONSTANTS_VMEC_ALGORITHM_CONSTANTS_H_
#define VMECPP_VMEC_VMEC_CONSTANTS_VMEC_ALGORITHM_CONSTANTS_H_

#include <array>
#include <cmath>
#include <numbers>

namespace vmecpp {

namespace vmec_algorithm_constants {

static constexpr int kSignOfJacobian = -1;

static constexpr double kMagneticFieldBlendingFactor = 0.05;

static constexpr double kVacuumPermeability = 4.0e-7 * M_PI;

static constexpr double kIonLarmorRadiusCoefficient = 3.2e-3;

static constexpr double kLambdaPreconditionerDampingFactor = 2.0;

static constexpr double kLambdaPreconditionerZeroGuard = -1.0e-10;

static constexpr double kToroidalNormalizationFactor = 2.0 * M_PI;

static constexpr double kToroidalVolumeFactor = 4.0 * M_PI * M_PI;

static constexpr double kConstraintScalingFactor = 1.0 / std::numbers::sqrt2;

static constexpr double kDefaultForceTolerance = 1.0e-10;

static constexpr double kFastConvergenceThreshold = 1.0e-6;

static constexpr double kVacuumPressureThreshold = 1.0e-3;

static constexpr double kCurrentScalingFactor = 1.0e-6;

static constexpr double kForceResidualThreshold = 1.0e-2;

static constexpr double kTangentialEpsilon = 1.0e-15;

static constexpr int kDefaultMaxIterations = 100;

static constexpr int kDefaultMultigridIterations = 500;

static constexpr int kMinIterationsForM1Constraint = 2;

static constexpr int kMaxIterationDeltaForEdgeForces = 50;

static constexpr int kJacobianIterationThreshold = 75;

static constexpr int kPreconditionerUpdateInterval = 25;

static constexpr int kNDamp = 10;

static constexpr int kDefaultRadialResolution = 31;

static constexpr int kMagneticAxisGridPoints = 61;

static constexpr int kStringBufferSize = 30;

static constexpr double kGeneralScalingFactor = 0.9;

static constexpr double kJacobianScaling25 = 0.98;
static constexpr double kJacobianScaling50 = 0.96;

static constexpr double kEdgePedestalFactor = 0.05;

static constexpr double kLambdaHighMDampingMaxPower = 8.0;

static constexpr double kLambdaHighMDampingReferenceM = 16.0;

static constexpr double kEigenvalueAvoidanceFactor = -1.0e-10;

static constexpr double kVacuumFrequencyLow = 0.1;
static constexpr double kVacuumFrequencyHigh = 1.0e11;

static constexpr int kEvenParity = 0;

static constexpr int kOddParity = 1;

static constexpr std::array<double, 10> kGaussLegendreWeights10 = {
0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
0.0666713443086881};

static constexpr std::array<double, 10> kGaussLegendreAbscissae10 = {
-0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
-0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
0.4333953941292472,  0.6794095682990244,  0.8650633666889845,
0.9739065285171717};

}

}

#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vmecpp {

using RowMatrixXd =
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline void SumOverThreads(const double *contribution, int width, int thread_id,
int num_threads, double *m_slots, double *m_total) {
double *row = m_slots + thread_id * width;
for (int i = 0; i < width; ++i) {
row[i] = contribution[i];
}
#ifdef _OPENMP
#pragma omp barrier
#pragma omp single
#endif
{
for (int i = 0; i < width; ++i) {
m_total[i] = 0.0;
}
for (int t = 0; t < num_threads; ++t) {
const double *slot = m_slots + t * width;
for (int i = 0; i < width; ++i) {
m_total[i] += slot[i];
}
}
}
}

inline Eigen::VectorXd ToEigenVector(const std::vector<double> &v) {
return Eigen::Map<const Eigen::VectorXd>(v.data(),
static_cast<Eigen::Index>(v.size()));
}

inline Eigen::VectorXi ToEigenVector(const std::vector<int> &v) {
return Eigen::Map<const Eigen::VectorXi>(v.data(),
static_cast<Eigen::Index>(v.size()));
}

inline vmecpp::RowMatrixXd ToEigenMatrix(const std::vector<double> &v,
Eigen::Index size1,
Eigen::Index size2) {
return Eigen::Map<const vmecpp::RowMatrixXd>(v.data(), size1, size2);
}

inline vmecpp::RowMatrixXd ToEigenMatrix(
const std::vector<std::vector<double>> &v) {
const std::size_t outer_size = v.size();
CHECK_GT(outer_size, 0u);
const std::size_t inner_size = v[0].size();
for (const auto &row : v) {
CHECK_EQ(row.size(), inner_size);
}

vmecpp::RowMatrixXd m(outer_size, inner_size);

for (int i = 0; i < m.rows(); ++i) {
for (int j = 0; j < m.cols(); ++j) {
m(i, j) = v[i][j];
}
}

return m;
}

enum class VmecCheckpoint : std::uint8_t {
NONE = 0,

RADIAL_PROFILES_EVAL,
SPECTRAL_CONSTRAINT,
SETUP_INITIAL_STATE,

FOURIER_GEOMETRY_TO_START_WITH,
INV_DFT_GEOMETRY,
JACOBIAN,
METRIC,
VOLUME,
B_CONTRA,
B_CO,
ENERGY,

RADIAL_FORCE_BALANCE,
HYBRID_LAMBDA_FORCE,
REALSPACE_FORCES,
UPDATE_RADIAL_PRECONDITIONER,
UPDATE_FORCE_NORMS,
UPDATE_TCON,
ALIAS,
FWD_DFT_FORCES,

VAC1_VACUUM,
VAC1_SURFACE,
VAC1_BEXTERN,
VAC1_ANALYT,
VAC1_GREENF,
VAC1_FOURP,
VAC1_FOURI_SYMM,
VAC1_FOURI_KV_DFT,
VAC1_FOURI_KU_DFT,
VAC1_SOLVER,
VAC1_BSQVAC,

RBSQ,

PHYSICAL_FORCES,
INVARIANT_RESIDUALS,
APPLY_M1_PRECONDITIONER,
ASSEMBLE_RZ_PRECONDITIONER,
APPLY_RADIAL_PRECONDITIONER,
PRECONDITIONED_RESIDUALS,

PRINTOUT,
EVOLVE,

INTERP,

BCOVAR_FILEOUT,
BSS,
LOWPASS_BCOVARIANT,
EXTRAPOLATE_BSUBS,
JXBOUT,
MERCIER,
THREED1_FIRST_TABLE,
THREED1_GEOMAG,
THREED1_VOLUMETRICS,
THREED1_AXIS,
THREED1_BETAS,
THREED1_SHAFRANOV_INTEGRALS
};

enum class VmecStatus : std::uint8_t {

NORMAL_TERMINATION = 0,
BAD_JACOBIAN = 1,
JACOBIAN_75_TIMES_BAD = 4,

UNRECOVERABLE_ERROR = 5,

SUCCESSFUL_TERMINATION = 11
};

enum class VacuumPressureState : std::int8_t {

kOff = -1,

kInitializing = 0,

kInitialized = 1,

kActive = 2,

kSettled = 3
};

int VmecStatusCode(const VmecStatus vmec_status);

std::string VmecStatusAsString(const VmecStatus vmec_status);

static constexpr double MU_0 = 4.0e-7 * M_PI;

int signum(int x);

void TridiagonalSolveSerial(std::span<double> m_a, std::span<double> m_d,
std::span<double> m_b, double *m_c_data,
int c_stride, int jMin, int jMax, int nRHS);

void TridiagonalSolveOpenMP(
std::vector<double> &m_ar, std::vector<double> &m_dr,
std::vector<double> &m_br, std::vector<std::span<double>> &m_cr,
std::vector<double> &m_az, std::vector<double> &m_dz,
std::vector<double> &m_bz, std::vector<std::span<double>> &m_cz,
const std::vector<int> &jMin, int jMax, int mnmax, int nRHS,
std::vector<std::mutex> &m_mutices, int ncpu, int myid, int nsMinF,
int nsMaxF, std::vector<double> &m_handover_ar,
std::vector<std::vector<double>> &m_handover_cr,
std::vector<double> &m_handover_az,
std::vector<std::vector<double>> &m_handover_cz);

int vmec_adjust_num_threads(int max_threads, int num_surfaces_to_distribute);

int vmec_adjust_vacuum_num_threads(int max_threads, int n_znt);

}

#endif

// header: vmecpp/common/vmec_indata/vmec_indata.h

#ifndef VMECPP_COMMON_VMEC_INDATA_VMEC_INDATA_H_
#define VMECPP_COMMON_VMEC_INDATA_VMEC_INDATA_H_

#include <Eigen/Dense>
#include <filesystem>
#include <string>

#include "H5Cpp.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace vmecpp {

static constexpr int kNsDefault = 31;

static constexpr double kFTolDefault = 1.0e-10;

static constexpr int kNIterDefault = 100;

enum class FreeBoundaryMethod : std::uint8_t {

NESTOR,

ONLY_COILS,

BIEST
};

int FreeBoundaryMethodCode(FreeBoundaryMethod free_boundary_method);
absl::StatusOr<FreeBoundaryMethod> FreeBoundaryMethodFromString(
const std::string& free_boundary_method_string);
std::string ToString(FreeBoundaryMethod free_boundary_method);

enum class IterationStyle : std::uint8_t {

VMEC_8_52,

PARVMEC
};

int IterationStyleCode(IterationStyle iteration_style);
absl::StatusOr<IterationStyle> IterationStyleFromString(
const std::string& iteration_style_string);
std::string ToString(IterationStyle iteration_style);

class VmecINDATA {
public:

bool lasym;

int nfp;

int mpol;

int ntor;

int mpol_geometry;
int ntor_geometry;

int ntheta;

int nzeta;

Eigen::VectorXi ns_array;

Eigen::VectorXd ftol_array;

Eigen::VectorXi niter_array;

double phiedge;

int ncurr;

std::string pmass_type;

Eigen::VectorXd am;

Eigen::VectorXd am_aux_s;

Eigen::VectorXd am_aux_f;

double pres_scale;

double gamma;

double spres_ped;

std::string piota_type;

Eigen::VectorXd ai;

Eigen::VectorXd ai_aux_s;

Eigen::VectorXd ai_aux_f;

std::string pcurr_type;

Eigen::VectorXd ac;

Eigen::VectorXd ac_aux_s;

Eigen::VectorXd ac_aux_f;

double curtor;

double bloat;

bool lfreeb;

std::string mgrid_file;

Eigen::VectorXd extcur;

int nvacskip;

int signgs;

FreeBoundaryMethod free_boundary_method;

int nstep;

Eigen::VectorXd aphi;

double delt;

double tcon0;

bool lforbal;

IterationStyle iteration_style;

bool return_outputs_even_if_not_converged;

Eigen::VectorXd raxis_c;

Eigen::VectorXd zaxis_s;

std::optional<Eigen::VectorXd> raxis_s;

std::optional<Eigen::VectorXd> zaxis_c;

RowMatrixXd rbc;

RowMatrixXd zbs;

std::optional<RowMatrixXd> rbs;

std::optional<RowMatrixXd> zbc;

VmecINDATA();

bool operator==(const VmecINDATA&) const = default;
bool operator!=(const VmecINDATA& o) const { return !(*this == o); }

void SetMpolNtor(int new_mpol, int new_ntor);

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(VmecINDATA& m_obj, H5::H5File& from_file);

static VmecINDATA FromFile(
const std::filesystem::path& indata_json_file_path);
static absl::StatusOr<VmecINDATA> FromJson(const std::string& indata_json);

absl::StatusOr<std::string> ToJson() const;
std::string ToJsonOrException() const;

VmecINDATA Copy() const;
};

absl::Status IsConsistent(const VmecINDATA& vmec_indata,
bool enable_info_messages);

}

#endif

namespace vmecpp {

class Sizes {
public:

explicit Sizes(const VmecINDATA& id);

Sizes(bool lasym, int nfp, int mpol, int ntor, int ntheta, int nzeta,
int mpol_geometry = -1, int ntor_geometry = -1);

bool lasym;

int nfp;

int mpol;

int ntor;

int mpolGeometry;
int ntorGeometry;

int ntheta;

int nZeta;

bool lthreed;

int num_basis;

int nThetaEven;
int nThetaReduced;
int nThetaEff;

int nZnT;

Eigen::VectorXd wInt;

int mnsize;

int mnmax;

int mnyq2;

int nnyq2;

int mnyq;

int nnyq;

int mnmax_nyq;

private:
void computeDerivedSizes();
};

}

#endif

namespace vmecpp {

struct FourierBasisFastPoloidalLayout {

static int PoloidalBasisIndex(int m, int l, int num_m, int num_l) {
(void)num_m;
return m * num_l + l;
}

static int ToroidalBasisIndex(int n, int k, int num_n, int num_k) {
(void)num_k;
return k * num_n + n;
}

static int ProductIndex(int m, int n, int m_size, int n_size) {
(void)m_size;
return m * (n_size + 1) + n;
}
};

struct FourierBasisFastToroidalLayout {
static int PoloidalBasisIndex(int m, int l, int num_m, int num_l) {
(void)num_l;
return l * num_m + m;
}
static int ToroidalBasisIndex(int n, int k, int num_n, int num_k) {
(void)num_n;
return n * num_k + k;
}
static int ProductIndex(int m, int n, int m_size, int n_size) {
(void)n_size;
return n * m_size + m;
}
};

template <class Layout>
class FourierBasis {
public:
explicit FourierBasis(const Sizes* s);

Eigen::VectorXd mscale;

Eigen::VectorXd nscale;

Eigen::VectorXd cosmu;

Eigen::VectorXd sinmu;

Eigen::VectorXd cosmum;

Eigen::VectorXd sinmum;

Eigen::VectorXd cosmui;

Eigen::VectorXd sinmui;

Eigen::VectorXd cosmumi;

Eigen::VectorXd sinmumi;

Eigen::VectorXd cosnv;

Eigen::VectorXd sinnv;

Eigen::VectorXd cosnvn;

Eigen::VectorXd sinnvn;

int cos_to_cc_ss(const std::span<const double> fcCos,
std::span<double> m_fcCC, std::span<double> m_fcSS,
int n_size, int m_size) const;

int sin_to_sc_cs(const std::span<const double> fcSin,
std::span<double> m_fcSC, std::span<double> m_fcCS,
int n_size, int m_size) const;

int cc_ss_to_cos(const std::span<const double> fcCC,
const std::span<const double> fcSS,
std::span<double> m_fcCos, int n_size, int m_size) const;

int sc_cs_to_sin(const std::span<const double> fcSC,
const std::span<const double> fcCS,
std::span<double> m_fcSin, int n_size, int m_size) const;

int mnIdx(int m, int n) const;
int mnMax(int m_size, int n_size) const;
void computeConversionIndices(Eigen::VectorXi& m_xm, Eigen::VectorXi& m_xn,
int n_size, int m_size, int nfp) const;

Eigen::VectorXi xm;

Eigen::VectorXi xn;

Eigen::VectorXi xm_nyq;

Eigen::VectorXi xn_nyq;

private:
const Sizes& s_;

void computeFourierBasis(int nfp);
};

using FourierBasisFastPoloidal = FourierBasis<FourierBasisFastPoloidalLayout>;
using FourierBasisFastToroidal = FourierBasis<FourierBasisFastToroidalLayout>;

struct SurfaceFourierGeometry {

std::span<const double> rmncc;

std::span<const double> rmnss;

std::span<const double> rmnsc;

std::span<const double> rmncs;

std::span<const double> zmnsc;

std::span<const double> zmncs;

std::span<const double> zmncc;

std::span<const double> zmnss;
};

double SpectralWidth(const SurfaceFourierGeometry& geometry, const Sizes& sizes,
std::span<const double> mscale,
std::span<const double> nscale, int p = 4, int q = 1);

}

#endif

#include <cmath>
#include <numbers>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"

namespace vmecpp {

template <class Layout>
FourierBasis<Layout>::FourierBasis(const Sizes* s) : s_(*s) {
mscale.resize(s_.mnyq2 + 1);
nscale.resize(s_.nnyq2 + 1);

cosmu.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
sinmu.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
cosmum.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
sinmum.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
cosmui.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
sinmui.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
cosmumi.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
sinmumi.resize(s_.nThetaReduced * (s_.mnyq2 + 1));

cosnv.resize((s_.nnyq2 + 1) * s_.nZeta);
sinnv.resize((s_.nnyq2 + 1) * s_.nZeta);
cosnvn.resize((s_.nnyq2 + 1) * s_.nZeta);
sinnvn.resize((s_.nnyq2 + 1) * s_.nZeta);

computeFourierBasis(s_.nfp);

xm.resize(s_.mnmax);
xm.setZero();
xn.resize(s_.mnmax);
xn.setZero();

computeConversionIndices( xm,  xn, s_.ntor, s_.mpol, s_.nfp);

xm_nyq.resize(s_.mnmax_nyq);
xm_nyq.setZero();
xn_nyq.resize(s_.mnmax_nyq);
xn_nyq.setZero();

computeConversionIndices( xm_nyq,  xn_nyq, s_.nnyq,
s_.mnyq + 1, s_.nfp);
}

template <class Layout>
void FourierBasis<Layout>::computeFourierBasis(int nfp) {
static constexpr double kTwoPi = 2.0 * M_PI;

const double intNorm = 1.0 / (s_.nZeta * (s_.nThetaReduced - 1));

for (int m = 0; m < s_.mnyq2 + 1; ++m) {

if (m == 0) {
mscale[m] = 1.0;
} else {
mscale[m] = std::numbers::sqrt2;
}
}

for (int m = 0; m < s_.mnyq2 + 1; ++m) {
for (int l = 0; l < s_.nThetaReduced; ++l) {

const double theta = kTwoPi * l / s_.nThetaEven;
const int idx_ml =
Layout::PoloidalBasisIndex(m, l, s_.mnyq2 + 1, s_.nThetaReduced);

const double arg = m * theta;

cosmu[idx_ml] = std::cos(arg) * mscale[m];
sinmu[idx_ml] = std::sin(arg) * mscale[m];

cosmui[idx_ml] = cosmu[idx_ml] * intNorm;
sinmui[idx_ml] = sinmu[idx_ml] * intNorm;

if (l == 0 || l == s_.nThetaReduced - 1) {
cosmui[idx_ml] /= 2.0;
}

cosmum[idx_ml] = m * cosmu[idx_ml];
sinmum[idx_ml] = -m * sinmu[idx_ml];

cosmumi[idx_ml] = m * cosmui[idx_ml];
sinmumi[idx_ml] = -m * sinmui[idx_ml];
}
}

for (int n = 0; n < s_.nnyq2 + 1; ++n) {

if (n == 0) {
nscale[n] = 1.0;
} else {
nscale[n] = std::numbers::sqrt2;
}
}

for (int k = 0; k < s_.nZeta; ++k) {
const double zeta = kTwoPi * k / s_.nZeta;
for (int n = 0; n < s_.nnyq2 + 1; ++n) {
const int idx_kn =
Layout::ToroidalBasisIndex(n, k, s_.nnyq2 + 1, s_.nZeta);

const double arg = n * zeta;

cosnv[idx_kn] = std::cos(arg) * nscale[n];
sinnv[idx_kn] = std::sin(arg) * nscale[n];

cosnvn[idx_kn] = n * nfp * cosnv[idx_kn];
sinnvn[idx_kn] = -n * nfp * sinnv[idx_kn];
}
}
}

template <class Layout>
int FourierBasis<Layout>::cos_to_cc_ss(const std::span<const double> fcCos,
std::span<double> m_fcCC,
std::span<double> m_fcSS, int n_size,
int m_size) const {

int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

absl::c_fill_n(m_fcCC, m_size * (n_size + 1), 0);
absl::c_fill_n(m_fcSS, m_size * (n_size + 1), 0);

int mn = 0;

int m = 0;
for (int n = 0; n < n_size + 1; ++n) {
int abs_n = abs(n);

double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

double normedFC = basis_norm * fcCos[mn];

m_fcCC[Layout::ProductIndex(m, abs_n, m_size, n_size)] += normedFC;

mn++;
}

for (m = 1; m < m_size; ++m) {
for (int n = -n_size; n < n_size + 1; ++n) {
int abs_n = abs(n);
int sgn_n = signum(n);

double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

double normedFC = basis_norm * fcCos[mn];

m_fcCC[Layout::ProductIndex(m, abs_n, m_size, n_size)] += normedFC;
if (abs_n > 0) {
m_fcSS[Layout::ProductIndex(m, abs_n, m_size, n_size)] +=
sgn_n * normedFC;
}

mn++;
}
}

CHECK_EQ(mn, mnmax) << "counting error: mn=" << mn << " should be " << mnmax
<< " in cos_to_cc_ss";

return mnmax;
}

template <class Layout>
int FourierBasis<Layout>::sin_to_sc_cs(const std::span<const double> fcSin,
std::span<double> m_fcSC,
std::span<double> m_fcCS, int n_size,
int m_size) const {

int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

absl::c_fill_n(m_fcSC, m_size * (n_size + 1), 0);
absl::c_fill_n(m_fcCS, m_size * (n_size + 1), 0);

int mn = 1;

int m = 0;
for (int n = 1; n < n_size + 1; ++n) {
int abs_n = abs(n);
int sgn_n = signum(n);

double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

double normedFC = basis_norm * fcSin[mn];

m_fcCS[Layout::ProductIndex(m, abs_n, m_size, n_size)] = -sgn_n * normedFC;

mn++;
}

for (m = 1; m < m_size; ++m) {
for (int n = -n_size; n < n_size + 1; ++n) {
int abs_n = abs(n);
int sgn_n = signum(n);

double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

double normedFC = basis_norm * fcSin[mn];

m_fcSC[Layout::ProductIndex(m, abs_n, m_size, n_size)] += normedFC;
if (abs_n > 0) {
m_fcCS[Layout::ProductIndex(m, abs_n, m_size, n_size)] +=
-sgn_n * normedFC;
}

mn++;
}
}

CHECK_EQ(mn, mnmax) << "counting error: mn=" << mn << " should be " << mnmax
<< " in sin_to_sc_cs";

return mnmax;
}

template <class Layout>
int FourierBasis<Layout>::cc_ss_to_cos(const std::span<const double> fcCC,
const std::span<const double> fcSS,
std::span<double> m_fcCos, int n_size,
int m_size) const {

int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

absl::c_fill_n(m_fcCos, mnmax, 0);

int mn = 0;

int m = 0;
for (int n = 0; n < n_size + 1; ++n) {
double basis_norm = 1.0 / (mscale[m] * nscale[n]);

m_fcCos[mn] = fcCC[Layout::ProductIndex(m, n, m_size, n_size)] / basis_norm;

mn++;
}

for (m = 1; m < m_size; ++m) {
for (int n = -n_size; n < n_size + 1; ++n) {
int abs_n = abs(n);
int sgn_n = signum(n);

double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

if (abs_n == 0) {
m_fcCos[mn] =
fcCC[Layout::ProductIndex(m, abs_n, m_size, n_size)] / basis_norm;
} else {
double raw_cc = fcCC[Layout::ProductIndex(m, abs_n, m_size, n_size)];
double raw_ss = fcSS[Layout::ProductIndex(m, abs_n, m_size, n_size)];
m_fcCos[mn] = 0.5 * (raw_cc + sgn_n * raw_ss) / basis_norm;
}

mn++;
}
}

CHECK_EQ(mn, mnmax) << "counting error: mn=" << mn << " should be " << mnmax
<< " in cc_ss_to_cos";

return mnmax;
}

template <class Layout>
int FourierBasis<Layout>::sc_cs_to_sin(const std::span<const double> fcSC,
const std::span<const double> fcCS,
std::span<double> m_fcSin, int n_size,
int m_size) const {

int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

absl::c_fill_n(m_fcSin, mnmax, 0);

int mn = 1;

int m = 0;
for (int n = 1; n < n_size + 1; ++n) {
double basis_norm = 1.0 / (mscale[m] * nscale[n]);

m_fcSin[mn] =
-fcCS[Layout::ProductIndex(m, n, m_size, n_size)] / basis_norm;

mn++;
}

for (m = 1; m < m_size; ++m) {
for (int n = -n_size; n < n_size + 1; ++n) {
int abs_n = abs(n);
int sgn_n = signum(n);

double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

if (abs_n == 0) {
m_fcSin[mn] =
fcSC[Layout::ProductIndex(m, abs_n, m_size, n_size)] / basis_norm;
} else {
double raw_sc = fcSC[Layout::ProductIndex(m, abs_n, m_size, n_size)];
double raw_cs = fcCS[Layout::ProductIndex(m, abs_n, m_size, n_size)];
m_fcSin[mn] = 0.5 * (raw_sc - sgn_n * raw_cs) / basis_norm;
}

mn++;
}
}

CHECK_EQ(mn, mnmax) << "counting error: mn=" << mn << " should be " << mnmax
<< " in sc_cs_to_sin";

return mnmax;
}

template <class Layout>
int FourierBasis<Layout>::mnIdx(int m, int n) const {
if (m == 0) {
CHECK_GE(n, 0) << "no mn index available for n < 0";
return n;
} else {
return (s_.ntor + 1) + (m - 1) * (2 * s_.ntor + 1) + (n + s_.ntor);
}
}

template <class Layout>
int FourierBasis<Layout>::mnMax(int m_size, int n_size) const {

int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

return mnmax;
}

template <class Layout>
void FourierBasis<Layout>::computeConversionIndices(Eigen::VectorXi& m_xm,
Eigen::VectorXi& m_xn,
int n_size, int m_size,
int nfp) const {
const int mnmax = mnMax(m_size, n_size);
int mn = 0;

int m = 0;
for (int n = 0; n < n_size + 1; ++n) {
m_xm[mn] = m;
m_xn[mn] = n * nfp;
mn++;
}

for (m = 1; m < m_size; ++m) {
for (int n = -n_size; n < n_size + 1; ++n) {
m_xm[mn] = m;
m_xn[mn] = n * nfp;
mn++;
}
}

CHECK_EQ(mn, mnmax) << "counting error: mn=" << mn << " should be " << mnmax;
}

template class FourierBasis<FourierBasisFastPoloidalLayout>;
template class FourierBasis<FourierBasisFastToroidalLayout>;

double SpectralWidth(const SurfaceFourierGeometry& geometry, const Sizes& sizes,
std::span<const double> mscale,
std::span<const double> nscale, const int p, const int q) {
double spectral_width_numerator = 0.0;
double spectral_width_denominator = 0.0;

for (int m = 1; m < sizes.mpol; ++m) {
for (int n = 0; n < sizes.ntor + 1; ++n) {
const int fourier_index = m * (sizes.ntor + 1) + n;

const double basis_norm = mscale[m] * nscale[n];

Eigen::Vector4d r_coefficients = Eigen::Vector4d::Zero();
Eigen::Vector4d z_coefficients = Eigen::Vector4d::Zero();
int basis_dimension = 0;

r_coefficients[basis_dimension] = geometry.rmncc[fourier_index];
z_coefficients[basis_dimension] = geometry.zmnsc[fourier_index];
basis_dimension++;

if (sizes.lthreed) {
if (m == 1) {
const double r_plus = geometry.rmnss[fourier_index];
const double r_minus = geometry.zmncs[fourier_index];

r_coefficients[basis_dimension] = r_plus + r_minus;

z_coefficients[basis_dimension] = r_plus - r_minus;
} else {
r_coefficients[basis_dimension] = geometry.rmnss[fourier_index];
z_coefficients[basis_dimension] = geometry.zmncs[fourier_index];
}
basis_dimension++;
}
if (sizes.lasym) {
if (m == 1) {
const double r_plus = geometry.rmnsc[fourier_index];
const double r_minus = geometry.zmncc[fourier_index];

r_coefficients[basis_dimension] = r_plus + r_minus;

z_coefficients[basis_dimension] = r_plus - r_minus;
} else {
r_coefficients[basis_dimension] = geometry.rmnsc[fourier_index];
z_coefficients[basis_dimension] = geometry.zmncc[fourier_index];
}
basis_dimension++;
}

if (sizes.lasym && sizes.lthreed) {
r_coefficients[basis_dimension] = geometry.rmncs[fourier_index];
z_coefficients[basis_dimension] = geometry.zmnss[fourier_index];
basis_dimension++;
}

double coefficient_norm =
r_coefficients.head(basis_dimension).squaredNorm() +
z_coefficients.head(basis_dimension).squaredNorm();
coefficient_norm *= basis_norm * basis_norm;

spectral_width_numerator += coefficient_norm * std::pow(m, p + q);
spectral_width_denominator += coefficient_norm * std::pow(m, p);
}
}

return spectral_width_numerator / spectral_width_denominator;
}

}

// source: vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.cc

// header: vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h

#ifndef VMECPP_COMMON_MAGNETIC_CONFIGURATION_LIB_MAGNETIC_CONFIGURATION_LIB_H_
#define VMECPP_COMMON_MAGNETIC_CONFIGURATION_LIB_MAGNETIC_CONFIGURATION_LIB_H_

#include <Eigen/Dense>
#include <filesystem>
#include <string>

#include "absl/status/statusor.h"

// header: vmecpp/common/magnetic_configuration_definition/magnetic_configuration.h

#ifndef VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_
#define VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_

#include <cstdint>
#include <list>
#include <string>

namespace magnetics {

struct PolygonFilament {

bool has_name_ = false;
std::string name_;

std::list<composed_types::Vector3d> vertices_;

bool has_name() const { return has_name_; }
const std::string& name() const { return name_; }
void set_name(const std::string& value) {
name_ = value;
has_name_ = true;
}
void set_name(std::string&& value) {
name_ = std::move(value);
has_name_ = true;
}
void clear_name() {
name_.clear();
has_name_ = false;
}

int vertices_size() const { return static_cast<int>(vertices_.size()); }
const composed_types::Vector3d& vertices(int index) const {
auto it = vertices_.cbegin();
std::advance(it, index);
return *it;
}
composed_types::Vector3d* mutable_vertices(int index) {
auto it = vertices_.begin();
std::advance(it, index);
return &(*it);
}
composed_types::Vector3d* add_vertices() {
vertices_.emplace_back();
auto it = vertices_.end();
--it;
return &(*it);
}
const std::list<composed_types::Vector3d>& vertices() const {
return vertices_;
}
std::list<composed_types::Vector3d>* mutable_vertices() { return &vertices_; }
void clear_vertices() { vertices_.clear(); }

void Clear() {
clear_name();
clear_vertices();
}
};

struct CircularFilament {

bool has_name_ = false;
std::string name_;

bool has_center_ = false;
composed_types::Vector3d center_;

bool has_normal_ = false;
composed_types::Vector3d normal_;

bool has_radius_ = false;
double radius_ = 0.0;

bool has_name() const { return has_name_; }
const std::string& name() const { return name_; }
void set_name(const std::string& value) {
name_ = value;
has_name_ = true;
}
void set_name(std::string&& value) {
name_ = std::move(value);
has_name_ = true;
}
void clear_name() {
name_.clear();
has_name_ = false;
}

bool has_center() const { return has_center_; }
const composed_types::Vector3d& center() const { return center_; }
composed_types::Vector3d* mutable_center() {
has_center_ = true;
return &center_;
}
void set_center(const composed_types::Vector3d& value) {
center_ = value;
has_center_ = true;
}
void clear_center() {
center_ = composed_types::Vector3d();
has_center_ = false;
}

bool has_normal() const { return has_normal_; }
const composed_types::Vector3d& normal() const { return normal_; }
composed_types::Vector3d* mutable_normal() {
has_normal_ = true;
return &normal_;
}
void set_normal(const composed_types::Vector3d& value) {
normal_ = value;
has_normal_ = true;
}
void clear_normal() {
normal_ = composed_types::Vector3d();
has_normal_ = false;
}

bool has_radius() const { return has_radius_; }
double radius() const { return radius_; }
void set_radius(double value) {
radius_ = value;
has_radius_ = true;
}
void clear_radius() {
radius_ = 0.0;
has_radius_ = false;
}

void Clear() {
clear_name();
clear_center();
clear_normal();
clear_radius();
}
};

struct InfiniteStraightFilament {

bool has_name_ = false;
std::string name_;

bool has_origin_ = false;
composed_types::Vector3d origin_;

bool has_direction_ = false;
composed_types::Vector3d direction_;

bool has_name() const { return has_name_; }
const std::string& name() const { return name_; }
void set_name(const std::string& value) {
name_ = value;
has_name_ = true;
}
void set_name(std::string&& value) {
name_ = std::move(value);
has_name_ = true;
}
void clear_name() {
name_.clear();
has_name_ = false;
}

bool has_origin() const { return has_origin_; }
const composed_types::Vector3d& origin() const { return origin_; }
composed_types::Vector3d* mutable_origin() {
has_origin_ = true;
return &origin_;
}
void set_origin(const composed_types::Vector3d& value) {
origin_ = value;
has_origin_ = true;
}
void clear_origin() {
origin_ = composed_types::Vector3d();
has_origin_ = false;
}

bool has_direction() const { return has_direction_; }
const composed_types::Vector3d& direction() const { return direction_; }
composed_types::Vector3d* mutable_direction() {
has_direction_ = true;
return &direction_;
}
void set_direction(const composed_types::Vector3d& value) {
direction_ = value;
has_direction_ = true;
}
void clear_direction() {
direction_ = composed_types::Vector3d();
has_direction_ = false;
}

void Clear() {
clear_name();
clear_origin();
clear_direction();
}
};

struct CurrentCarrier {

enum TypeCase : std::uint8_t {
kInfiniteStraightFilament = 1,
kCircularFilament = 2,
kPolygonFilament = 3,
kFourierFilament = 4,
kTypeNotSet = 0
};

private:
TypeCase type_case_ = kTypeNotSet;

union {
InfiniteStraightFilament infinite_straight_filament_;
CircularFilament circular_filament_;
PolygonFilament polygon_filament_;
};

public:
CurrentCarrier() : type_case_(kTypeNotSet) {}

~CurrentCarrier() { Clear(); }

CurrentCarrier(const CurrentCarrier& other) : type_case_(kTypeNotSet) {
switch (other.type_case_) {
case kInfiniteStraightFilament: {
type_case_ = kInfiniteStraightFilament;
std::construct_at(std::addressof(infinite_straight_filament_),
other.infinite_straight_filament_);
} break;
case kCircularFilament: {
type_case_ = kCircularFilament;
std::construct_at(std::addressof(circular_filament_),
other.circular_filament_);
} break;
case kPolygonFilament: {
type_case_ = kPolygonFilament;
std::construct_at(std::addressof(polygon_filament_),
other.polygon_filament_);
} break;
default:
type_case_ = kTypeNotSet;
break;
}
}

CurrentCarrier(CurrentCarrier&& other) noexcept : type_case_(kTypeNotSet) {
switch (other.type_case_) {
case kInfiniteStraightFilament: {
type_case_ = kInfiniteStraightFilament;
std::construct_at(std::addressof(infinite_straight_filament_),
std::move(other.infinite_straight_filament_));
} break;
case kCircularFilament: {
type_case_ = kCircularFilament;
std::construct_at(std::addressof(circular_filament_),
std::move(other.circular_filament_));
} break;
case kPolygonFilament: {
type_case_ = kPolygonFilament;
std::construct_at(std::addressof(polygon_filament_),
std::move(other.polygon_filament_));
} break;
default:
type_case_ = kTypeNotSet;
break;
}
other.Clear();
}

CurrentCarrier& operator=(const CurrentCarrier& other) {
if (this != &other) {
Clear();
switch (other.type_case_) {
case kInfiniteStraightFilament: {
type_case_ = kInfiniteStraightFilament;
std::construct_at(std::addressof(infinite_straight_filament_),
other.infinite_straight_filament_);
} break;
case kCircularFilament: {
type_case_ = kCircularFilament;
std::construct_at(std::addressof(circular_filament_),
other.circular_filament_);
} break;
case kPolygonFilament: {
type_case_ = kPolygonFilament;
std::construct_at(std::addressof(polygon_filament_),
other.polygon_filament_);
} break;
default:
type_case_ = kTypeNotSet;
break;
}
}
return *this;
}

CurrentCarrier& operator=(CurrentCarrier&& other) noexcept {
if (this != &other) {
Clear();
switch (other.type_case_) {
case kInfiniteStraightFilament: {
type_case_ = kInfiniteStraightFilament;
std::construct_at(std::addressof(infinite_straight_filament_),
std::move(other.infinite_straight_filament_));
} break;
case kCircularFilament: {
type_case_ = kCircularFilament;
std::construct_at(std::addressof(circular_filament_),
std::move(other.circular_filament_));
} break;
case kPolygonFilament: {
type_case_ = kPolygonFilament;
std::construct_at(std::addressof(polygon_filament_),
std::move(other.polygon_filament_));
} break;
default:
type_case_ = kTypeNotSet;
break;
}
other.Clear();
}
return *this;
}

void Clear() {
switch (type_case_) {
case kInfiniteStraightFilament:
infinite_straight_filament_.~InfiniteStraightFilament();
break;
case kCircularFilament:
circular_filament_.~CircularFilament();
break;
case kPolygonFilament:
polygon_filament_.~PolygonFilament();
break;
default:
break;
}
type_case_ = kTypeNotSet;
}

bool has_infinite_straight_filament() const {
return type_case_ == kInfiniteStraightFilament;
}
const InfiniteStraightFilament& infinite_straight_filament() const {
return infinite_straight_filament_;
}
InfiniteStraightFilament* mutable_infinite_straight_filament() {
if (type_case_ != kInfiniteStraightFilament) {
Clear();
type_case_ = kInfiniteStraightFilament;
std::construct_at(std::addressof(infinite_straight_filament_));
}
return &infinite_straight_filament_;
}
void set_infinite_straight_filament(const InfiniteStraightFilament& value) {
Clear();
type_case_ = kInfiniteStraightFilament;
std::construct_at(std::addressof(infinite_straight_filament_), value);
}

bool has_circular_filament() const { return type_case_ == kCircularFilament; }
const CircularFilament& circular_filament() const {
return circular_filament_;
}
CircularFilament* mutable_circular_filament() {
if (type_case_ != kCircularFilament) {
Clear();
type_case_ = kCircularFilament;
std::construct_at(std::addressof(circular_filament_));
}
return &circular_filament_;
}
void set_circular_filament(const CircularFilament& value) {
Clear();
type_case_ = kCircularFilament;
std::construct_at(std::addressof(circular_filament_), value);
}

bool has_polygon_filament() const { return type_case_ == kPolygonFilament; }
const PolygonFilament& polygon_filament() const { return polygon_filament_; }
PolygonFilament* mutable_polygon_filament() {
if (type_case_ != kPolygonFilament) {
Clear();
type_case_ = kPolygonFilament;
std::construct_at(std::addressof(polygon_filament_));
}
return &polygon_filament_;
}
void set_polygon_filament(const PolygonFilament& value) {
Clear();
type_case_ = kPolygonFilament;
std::construct_at(std::addressof(polygon_filament_), value);
}

TypeCase type_case() const { return type_case_; }
};

struct Coil {

bool has_name_ = false;
std::string name_;

bool has_num_windings_ = false;
double num_windings_ = 0.0;

std::list<CurrentCarrier> current_carriers_;

bool has_name() const { return has_name_; }
const std::string& name() const { return name_; }
void set_name(const std::string& value) {
name_ = value;
has_name_ = true;
}
void set_name(std::string&& value) {
name_ = std::move(value);
has_name_ = true;
}
void clear_name() {
name_.clear();
has_name_ = false;
}

bool has_num_windings() const { return has_num_windings_; }
double num_windings() const { return num_windings_; }
void set_num_windings(double value) {
num_windings_ = value;
has_num_windings_ = true;
}
void clear_num_windings() {
num_windings_ = 0.0;
has_num_windings_ = false;
}

int current_carriers_size() const {
return static_cast<int>(current_carriers_.size());
}
const CurrentCarrier& current_carriers(int index) const {
auto it = current_carriers_.cbegin();

std::advance(it, index);
return *it;
}
CurrentCarrier* mutable_current_carriers(int index) {
auto it = current_carriers_.begin();

std::advance(it, index);
return &(*it);
}
CurrentCarrier* add_current_carriers() {
current_carriers_.emplace_back();
auto it = current_carriers_.end();

--it;
return &(*it);
}
const std::list<CurrentCarrier>& current_carriers() const {
return current_carriers_;
}
std::list<CurrentCarrier>* mutable_current_carriers() {
return &current_carriers_;
}
void clear_current_carriers() { current_carriers_.clear(); }

void Clear() {
clear_name();
clear_num_windings();
clear_current_carriers();
}
};

struct SerialCircuit {

bool has_name_ = false;
std::string name_;

bool has_current_ = false;
double current_ = 0.0;

std::list<Coil> coils_;

bool has_name() const { return has_name_; }
const std::string& name() const { return name_; }
void set_name(const std::string& value) {
name_ = value;
has_name_ = true;
}
void set_name(std::string&& value) {
name_ = std::move(value);
has_name_ = true;
}
void clear_name() {
name_.clear();
has_name_ = false;
}

bool has_current() const { return has_current_; }
double current() const { return current_; }
void set_current(double value) {
current_ = value;
has_current_ = true;
}
void clear_current() {
current_ = 0.0;
has_current_ = false;
}

int coils_size() const { return static_cast<int>(coils_.size()); }
const Coil& coils(int index) const {
auto it = coils_.cbegin();

std::advance(it, index);
return *it;
}
Coil* mutable_coils(int index) {
auto it = coils_.begin();

std::advance(it, index);
return &(*it);
}
Coil* add_coils() {
coils_.emplace_back();
auto it = coils_.end();
--it;
return &(*it);
}
const std::list<Coil>& coils() const { return coils_; }
std::list<Coil>* mutable_coils() { return &coils_; }
void clear_coils() { coils_.clear(); }

void Clear() {
clear_name();
clear_current();
clear_coils();
}
};

struct MagneticConfiguration {

bool has_name_ = false;
std::string name_;

bool has_num_field_periods_ = false;
int num_field_periods_ = 0;

std::list<SerialCircuit> serial_circuits_;

bool has_name() const { return has_name_; }
const std::string& name() const { return name_; }
void set_name(const std::string& value) {
name_ = value;
has_name_ = true;
}
void set_name(std::string&& value) {
name_ = std::move(value);
has_name_ = true;
}
void clear_name() {
name_.clear();
has_name_ = false;
}

bool has_num_field_periods() const { return has_num_field_periods_; }
int num_field_periods() const { return num_field_periods_; }
void set_num_field_periods(int value) {
num_field_periods_ = value;
has_num_field_periods_ = true;
}
void clear_num_field_periods() {
num_field_periods_ = 0;
has_num_field_periods_ = false;
}

int serial_circuits_size() const {
return static_cast<int>(serial_circuits_.size());
}
const SerialCircuit& serial_circuits(int index) const {
auto it = serial_circuits_.cbegin();

std::advance(it, index);
return *it;
}
SerialCircuit* mutable_serial_circuits(int index) {
auto it = serial_circuits_.begin();

std::advance(it, index);
return &(*it);
}
SerialCircuit* add_serial_circuits() {
serial_circuits_.emplace_back();
auto it = serial_circuits_.end();
--it;
return &(*it);
}
const std::list<SerialCircuit>& serial_circuits() const {
return serial_circuits_;
}
std::list<SerialCircuit>* mutable_serial_circuits() {
return &serial_circuits_;
}
void clear_serial_circuits() { serial_circuits_.clear(); }

void Clear() {
clear_name();
clear_num_field_periods();
clear_serial_circuits();
}
};

}

#endif

namespace magnetics {

absl::StatusOr<MagneticConfiguration> ImportMagneticConfigurationFromMakegrid(
const std::string& makegrid_coils);

absl::StatusOr<MagneticConfiguration> ImportMagneticConfigurationFromCoilsFile(
const std::filesystem::path& mgrid_coils_file);

absl::StatusOr<Eigen::VectorXd> GetCircuitCurrents(
const MagneticConfiguration& magnetic_configuration);

absl::Status SetCircuitCurrents(
const Eigen::VectorXd& circuit_currents,
MagneticConfiguration& m_magnetic_configuration);

absl::Status NumWindingsToCircuitCurrents(
MagneticConfiguration& m_magnetic_configuration);

absl::Status MoveRadially(double radial_step,
CircularFilament& m_circular_filament);

absl::Status MoveRadially(double radial_step,
PolygonFilament& m_polygon_filament);

absl::Status MoveRadially(double radial_step,
MagneticConfiguration& m_magnetic_configuration);

absl::Status IsInfiniteStraightFilamentFullyPopulated(
const InfiniteStraightFilament& infinite_straight_filament);

absl::Status IsCircularFilamentFullyPopulated(
const CircularFilament& circular_filament);

absl::Status IsPolygonFilamentFullyPopulated(
const PolygonFilament& polygon_filament);

absl::Status IsMagneticConfigurationFullyPopulated(
const MagneticConfiguration& magnetic_configuration);

void PrintInfiniteStraightFilament(
const InfiniteStraightFilament& infinite_straight_filament,
int indentation = 0);

void PrintCircularFilament(const CircularFilament& circular_filament,
int indentation = 0);

void PrintPolygonFilament(const PolygonFilament& polygon_filament,
int indentation = 0);

void PrintCurrentCarrier(const CurrentCarrier& current_carrier,
int indentation = 0);

void PrintCoil(const Coil& coil, int indentation = 0);

void PrintSerialCircuit(const SerialCircuit& serial_circuit,
int indentation = 0);

void PrintMagneticConfiguration(
const MagneticConfiguration& magnetic_configuration, int indentation = 0);

}

#endif

#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

namespace magnetics {

using composed_types::FourierCoefficient1D;
using composed_types::OrthonormalFrameAroundAxis;
using composed_types::Vector3d;

absl::Status ParseCurrentCarriers(
std::stringstream& m_makegrid_coils_ss,
MagneticConfiguration& m_magnetic_configuration) {
std::vector<int> coil_ids;

std::vector<double> x;
std::vector<double> y;
std::vector<double> z;
std::vector<double> w;

for (std::string raw_line; std::getline(m_makegrid_coils_ss, raw_line);) {
absl::string_view stripped_line = absl::StripAsciiWhitespace(raw_line);

if (absl::StartsWith(stripped_line, "mirror")) {

if (!(absl::EndsWithIgnoreCase(stripped_line, "NIL") ||
absl::EndsWithIgnoreCase(stripped_line, "NUL"))) {
return absl::InvalidArgumentError(absl::StrCat(
"The magnetic_configuration_lib only supports coilsets for which "
"the mirror option is deactivated. Please check your "
"coils file, should be 'mirror NUL' but found ",
stripped_line));
}
continue;
} else if (absl::StartsWith(stripped_line, "end")) {

return absl::OkStatus();
}

std::vector<std::string> line_parts = absl::StrSplit(
stripped_line, absl::ByAnyChar(" \t"), absl::SkipWhitespace());

const std::size_t num_line_parts = line_parts.size();

if (num_line_parts == 4 || num_line_parts == 6) {

x.push_back(std::stod(line_parts[0]));
y.push_back(std::stod(line_parts[1]));
z.push_back(std::stod(line_parts[2]));
w.push_back(std::stod(line_parts[3]));

if (num_line_parts == 6) {

int serial_circuit_id = std::stoi(line_parts[4]);
std::string current_carrier_name = line_parts[5];

std::vector<int>::iterator index_of_circuit_id =
std::find(coil_ids.begin(), coil_ids.end(), serial_circuit_id);
SerialCircuit* serial_circuit;
if (index_of_circuit_id != coil_ids.end()) {
const int serial_circuit_index =
static_cast<int>(index_of_circuit_id - coil_ids.begin());
serial_circuit = m_magnetic_configuration.mutable_serial_circuits(
serial_circuit_index);
} else {
serial_circuit = m_magnetic_configuration.add_serial_circuits();
serial_circuit->set_current(1.0);
coil_ids.push_back(serial_circuit_id);
}

Coil* coil = serial_circuit->add_coils();
coil->set_num_windings(w.at(0));

CurrentCarrier* current_carrier = coil->add_current_carriers();

if (x.size() == 1) {

CircularFilament* circular_filament =
current_carrier->mutable_circular_filament();
circular_filament->set_name(current_carrier_name);
circular_filament->set_radius(x.at(0));

Vector3d* center = circular_filament->mutable_center();
center->set_x(0.0);
center->set_y(0.0);
center->set_z(z.at(0));

Vector3d* normal = circular_filament->mutable_normal();
normal->set_x(0.0);
normal->set_y(0.0);
normal->set_z(1.0);

} else {

for (size_t i = 1; i < w.size() - 1; ++i) {
if (w.at(i) != coil->num_windings()) {

std::stringstream error_message;
error_message << "number of windings different at point " << i;
return absl::InvalidArgumentError(error_message.str());
}
}

PolygonFilament* polygon_filament =
current_carrier->mutable_polygon_filament();
polygon_filament->set_name(current_carrier_name);
for (size_t i = 0; i < x.size(); ++i) {
Vector3d* vertex = polygon_filament->add_vertices();
vertex->set_x(x.at(i));
vertex->set_y(y.at(i));
vertex->set_z(z.at(i));
}
}

x.clear();
y.clear();
z.clear();
w.clear();
}
} else {
std::stringstream error_message;
error_message << "cannot parse line: '" << stripped_line << "': has "
<< num_line_parts << " parts";
return absl::InvalidArgumentError(error_message.str());
}
}

return absl::ResourceExhaustedError(
"did not find 'end' line in makegrid_coils file");
}

absl::StatusOr<MagneticConfiguration> ImportMagneticConfigurationFromMakegrid(
const std::string& makegrid_coils) {
MagneticConfiguration magnetic_configuration;

std::stringstream makegrid_coils_ss(makegrid_coils);
for (std::string raw_line; std::getline(makegrid_coils_ss, raw_line);
) {
absl::string_view stripped_line = absl::StripAsciiWhitespace(raw_line);

if (absl::StartsWith(stripped_line, "periods")) {

std::vector<std::string> line_parts = absl::StrSplit(
stripped_line, absl::ByAnyChar(" \t"), absl::SkipWhitespace());
if (line_parts.size() != 2) {
std::stringstream error_message;
error_message << "expected number of field periods after 'periods', "
"but no second part was found on line '"
<< stripped_line << "'";
return absl::NotFoundError(error_message.str());
} else {
magnetic_configuration.set_num_field_periods(std::stoi(line_parts[1]));
}
} else if (absl::StartsWith(stripped_line, "begin filament")) {

absl::Status status =
ParseCurrentCarriers(makegrid_coils_ss, magnetic_configuration);
if (status != absl::OkStatus()) {

magnetic_configuration.Clear();
return status;
}
}
}

return magnetic_configuration;
}

absl::StatusOr<MagneticConfiguration> ImportMagneticConfigurationFromCoilsFile(
const std::filesystem::path& mgrid_coils_file) {
const auto maybe_coils_file_content = file_io::ReadFile(mgrid_coils_file);
if (!maybe_coils_file_content.ok()) {
return maybe_coils_file_content.status();
}
return ImportMagneticConfigurationFromMakegrid(*maybe_coils_file_content);
}

absl::StatusOr<Eigen::VectorXd> GetCircuitCurrents(
const MagneticConfiguration& magnetic_configuration) {
absl::Status status =
IsMagneticConfigurationFullyPopulated(magnetic_configuration);
if (!status.ok()) {
return status;
}

const int number_of_serial_circuits =
magnetic_configuration.serial_circuits_size();
Eigen::VectorXd circuit_currents(number_of_serial_circuits);
for (int i = 0; i < number_of_serial_circuits; ++i) {
circuit_currents[i] = magnetic_configuration.serial_circuits(i).current();
}

return circuit_currents;
}

absl::Status SetCircuitCurrents(
const Eigen::VectorXd& circuit_currents,
MagneticConfiguration& m_magnetic_configuration) {
const int number_of_serial_circuits =
m_magnetic_configuration.serial_circuits_size();
const Eigen::VectorXd::Index number_of_circuit_currents =
circuit_currents.size();
if (number_of_serial_circuits != number_of_circuit_currents) {
std::stringstream error_message;
error_message << "The number of circuit currents ("
<< number_of_circuit_currents << ") ";
error_message << "has to equal number of SerialCircuits ("
<< number_of_serial_circuits << ") ";
error_message << "in the given MagneticConfiguration.";
return absl::InvalidArgumentError(error_message.str());
}

for (int i = 0; i < number_of_serial_circuits; ++i) {
m_magnetic_configuration.mutable_serial_circuits(i)->set_current(
circuit_currents[i]);
}

return absl::OkStatus();
}

absl::Status NumWindingsToCircuitCurrents(
MagneticConfiguration& m_magnetic_configuration) {
const int num_serial_circuits =
m_magnetic_configuration.serial_circuits_size();
for (int idx_circuit = 0; idx_circuit < num_serial_circuits; ++idx_circuit) {
SerialCircuit* m_serial_circuit =
m_magnetic_configuration.mutable_serial_circuits(idx_circuit);
const int num_coils = m_serial_circuit->coils_size();

const double reference_num_windings =
m_serial_circuit->coils(0).num_windings();

const double current_times_num_windings =
m_serial_circuit->current() * reference_num_windings;
m_serial_circuit->set_current(current_times_num_windings);

for (int idx_coil = 0; idx_coil < num_coils; ++idx_coil) {
Coil* m_coil = m_serial_circuit->mutable_coils(idx_coil);
m_coil->set_num_windings(m_coil->num_windings() / reference_num_windings);
}
}

return absl::OkStatus();
}

absl::Status MoveRadially(double radial_step,
CircularFilament& m_circular_filament) {

const Vector3d& center = m_circular_filament.center();
if (center.x() != 0.0 || center.y() != 0.0) {
return absl::InvalidArgumentError(
"center has to be on origin in x and y to perform radial movement");
}

const Vector3d& normal = m_circular_filament.normal();
if (normal.x() != 0.0 || normal.y() != 0.0 || normal.z() == 0.0) {
return absl::InvalidArgumentError(
"normal has to be along z axis to perform radial movement");
}

m_circular_filament.set_radius(m_circular_filament.radius() + radial_step);

return absl::OkStatus();
}

absl::Status MoveRadially(double radial_step,
PolygonFilament& m_polygon_filament) {
int num_vertices = m_polygon_filament.vertices_size();
for (int i = 0; i < num_vertices; ++i) {
Vector3d* vertex = m_polygon_filament.mutable_vertices(i);
const double r =
std::sqrt(vertex->x() * vertex->x() + vertex->y() * vertex->y());
const double phi = std::atan2(vertex->y(), vertex->x());
vertex->set_x((r + radial_step) * std::cos(phi));
vertex->set_y((r + radial_step) * std::sin(phi));

}
return absl::OkStatus();
}

absl::Status MoveRadially(double radial_step,
MagneticConfiguration& m_magnetic_configuration) {
const int num_serial_circuits =
m_magnetic_configuration.serial_circuits_size();
for (int idx_circuit = 0; idx_circuit < num_serial_circuits; ++idx_circuit) {
SerialCircuit* m_serial_circuit =
m_magnetic_configuration.mutable_serial_circuits(idx_circuit);
const int num_coils = m_serial_circuit->coils_size();
for (int idx_coil = 0; idx_coil < num_coils; ++idx_coil) {
Coil* m_coil = m_serial_circuit->mutable_coils(idx_coil);
const int num_current_carriers = m_coil->current_carriers_size();
for (int idx_current_carrier = 0;
idx_current_carrier < num_current_carriers; ++idx_current_carrier) {
CurrentCarrier* m_current_carrier =
m_coil->mutable_current_carriers(idx_current_carrier);
switch (m_current_carrier->type_case()) {
case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
return absl::InvalidArgumentError(
"Cannot perform radial movement if an InfiniteStraightFilament "
"is present in the MagneticConfiguration");
case CurrentCarrier::TypeCase::kCircularFilament:
CHECK_OK(MoveRadially(
radial_step,
*(m_current_carrier->mutable_circular_filament())));
break;
case CurrentCarrier::TypeCase::kPolygonFilament:
CHECK_OK(MoveRadially(
radial_step, *(m_current_carrier->mutable_polygon_filament())));
break;
case CurrentCarrier::TypeCase::kFourierFilament:
return absl::InvalidArgumentError(
"Cannot perform radial movement if an FourierFilament is "
"present in the MagneticConfiguration");
case CurrentCarrier::TypeCase::kTypeNotSet:

break;
default:
std::stringstream error_message;
error_message << "current carrier type ";
error_message << m_current_carrier->type_case();
error_message << " not implemented yet.";
return absl::InvalidArgumentError(error_message.str());
}
}
}
}

return absl::OkStatus();
}

std::string CurrentCarrierIdentifier(
const InfiniteStraightFilament& infinite_straight_filament) {
std::stringstream current_carrier_identifier;
current_carrier_identifier << "InfiniteStraightFilament";
if (infinite_straight_filament.has_name()) {
current_carrier_identifier << " " << infinite_straight_filament.name();
}
return current_carrier_identifier.str();
}

std::string CurrentCarrierIdentifier(
const CircularFilament& circular_filament) {
std::stringstream current_carrier_identifier;
current_carrier_identifier << "CircularFilament";
if (circular_filament.has_name()) {
current_carrier_identifier << " " << circular_filament.name();
}
return current_carrier_identifier.str();
}

std::string CurrentCarrierIdentifier(const PolygonFilament& polygon_filament) {
std::stringstream current_carrier_identifier;
current_carrier_identifier << "PolygonFilament";
if (polygon_filament.has_name()) {
current_carrier_identifier << " " << polygon_filament.name();
}
return current_carrier_identifier.str();
}

absl::Status IsInfiniteStraightFilamentFullyPopulated(
const InfiniteStraightFilament& infinite_straight_filament) {
if (!infinite_straight_filament.has_origin()) {
std::stringstream error_message;
error_message << CurrentCarrierIdentifier(infinite_straight_filament);
error_message << " has no origin.";
return absl::NotFoundError(error_message.str());
} else {

const Vector3d& origin = infinite_straight_filament.origin();
absl::Status status = IsVector3dFullyPopulated(
origin, absl::StrCat("origin of ", CurrentCarrierIdentifier(
infinite_straight_filament)));
if (!status.ok()) {
return status;
}
}

if (!infinite_straight_filament.has_direction()) {
std::stringstream error_message;
error_message << CurrentCarrierIdentifier(infinite_straight_filament);
error_message << " has no direction.";
return absl::NotFoundError(error_message.str());
} else {

const Vector3d& direction = infinite_straight_filament.direction();
absl::Status status = IsVector3dFullyPopulated(
direction,
absl::StrCat("direction of ",
CurrentCarrierIdentifier(infinite_straight_filament)));
if (!status.ok()) {
return status;
}
}

return absl::OkStatus();
}

absl::Status IsCircularFilamentFullyPopulated(
const CircularFilament& circular_filament) {
if (!circular_filament.has_center()) {
std::stringstream error_message;
error_message << CurrentCarrierIdentifier(circular_filament);
error_message << " has no center.";
return absl::NotFoundError(error_message.str());
} else {

const Vector3d& center = circular_filament.center();
absl::Status status = IsVector3dFullyPopulated(
center, absl::StrCat("center of ",
CurrentCarrierIdentifier(circular_filament)));
if (!status.ok()) {
return status;
}
}

if (!circular_filament.has_normal()) {
std::stringstream error_message;
error_message << CurrentCarrierIdentifier(circular_filament);
error_message << " has no normal.";
return absl::NotFoundError(error_message.str());
} else {

const Vector3d& normal = circular_filament.normal();
absl::Status status = IsVector3dFullyPopulated(
normal, absl::StrCat("normal of ",
CurrentCarrierIdentifier(circular_filament)));
if (!status.ok()) {
return status;
}
}

if (!circular_filament.has_radius()) {
std::stringstream error_message;
error_message << CurrentCarrierIdentifier(circular_filament);
error_message << " has no radius.";
return absl::NotFoundError(error_message.str());
}

return absl::OkStatus();
}

absl::Status IsPolygonFilamentFullyPopulated(
const PolygonFilament& polygon_filament) {
if (polygon_filament.vertices_size() < 2) {
std::stringstream error_message;
error_message << CurrentCarrierIdentifier(polygon_filament);
error_message << " has too few vertices ("
<< polygon_filament.vertices_size() << "); need at least 2.";
return absl::NotFoundError(error_message.str());
}

for (int i = 0; i < polygon_filament.vertices_size(); ++i) {
const Vector3d& vertex = polygon_filament.vertices(i);
std::stringstream vertex_identifier;
vertex_identifier << "vertex[" << i << "]";
absl::Status status = IsVector3dFullyPopulated(
vertex, absl::StrCat(vertex_identifier.str(), " of ",
CurrentCarrierIdentifier(polygon_filament)));
if (!status.ok()) {
return status;
}
}

return absl::OkStatus();
}

absl::Status IsMagneticConfigurationFullyPopulated(
const MagneticConfiguration& magnetic_configuration) {
for (const SerialCircuit& serial_circuit :
magnetic_configuration.serial_circuits()) {
for (const Coil& coil : serial_circuit.coils()) {
for (const CurrentCarrier& current_carrier : coil.current_carriers()) {
absl::Status status;
switch (current_carrier.type_case()) {
case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
status = IsInfiniteStraightFilamentFullyPopulated(
current_carrier.infinite_straight_filament());
break;
case CurrentCarrier::TypeCase::kCircularFilament:
status = IsCircularFilamentFullyPopulated(
current_carrier.circular_filament());
break;
case CurrentCarrier::TypeCase::kPolygonFilament:
status = IsPolygonFilamentFullyPopulated(
current_carrier.polygon_filament());
break;
case CurrentCarrier::TypeCase::kTypeNotSet:

break;
default:
std::stringstream error_message;
error_message << "current carrier type ";
error_message << current_carrier.type_case();
error_message << " not implemented yet.";
status = absl::UnimplementedError(error_message.str());
}

if (!status.ok()) {
return status;
}
}
}
}

return absl::OkStatus();
}

void PrintInfiniteStraightFilament(
const InfiniteStraightFilament& infinite_straight_filament,
int indentation) {
std::string prefix;
for (int i = 0; i < indentation; ++i) {
prefix += " ";
}

std::cout << prefix << "InfiniteStraightFilament {" << '\n';

if (infinite_straight_filament.has_name()) {
std::cout << prefix << "  name: '" << infinite_straight_filament.name()
<< "'" << '\n';
} else {
std::cout << prefix << "  name: none" << '\n';
}

if (infinite_straight_filament.has_origin()) {
const Vector3d& origin = infinite_straight_filament.origin();
std::cout << prefix << "  origin: [" << origin.x() << ", " << origin.y()
<< ", " << origin.z() << "]" << '\n';
} else {
std::cout << prefix << "  origin: none" << '\n';
}

if (infinite_straight_filament.has_direction()) {
const Vector3d& direction = infinite_straight_filament.direction();
std::cout << prefix << "  direction: [" << direction.x() << ", "
<< direction.y() << ", " << direction.z() << "]" << '\n';
} else {
std::cout << prefix << "  direction: none" << '\n';
}

std::cout << prefix << "}" << '\n';
}

void PrintCircularFilament(const CircularFilament& circular_filament,
int indentation) {
std::string prefix;
for (int i = 0; i < indentation; ++i) {
prefix += " ";
}

std::cout << prefix << "CircularFilament {" << '\n';

if (circular_filament.has_name()) {
std::cout << prefix << "  name: '" << circular_filament.name() << "'"
<< '\n';
} else {
std::cout << prefix << "  name: none" << '\n';
}

if (circular_filament.has_center()) {
const Vector3d& center = circular_filament.center();
std::cout << prefix << "  center: [" << center.x() << ", " << center.y()
<< ", " << center.z() << "]" << '\n';
} else {
std::cout << prefix << "  center: none" << '\n';
}

if (circular_filament.has_normal()) {
const Vector3d& normal = circular_filament.normal();
std::cout << prefix << "  normal: [" << normal.x() << ", " << normal.y()
<< ", " << normal.z() << "]" << '\n';
} else {
std::cout << prefix << "  normal: none" << '\n';
}

if (circular_filament.has_radius()) {
const double radius = circular_filament.radius();
std::cout << prefix << "  radius: " << radius << '\n';
} else {
std::cout << prefix << "  radius: none" << '\n';
}

std::cout << prefix << "}" << '\n';
}

void PrintPolygonFilament(const PolygonFilament& polygon_filament,
int indentation) {
std::string prefix;
for (int i = 0; i < indentation; ++i) {
prefix += " ";
}

std::cout << prefix << "PolygonFilament {" << '\n';

if (polygon_filament.has_name()) {
std::cout << prefix << "  name: '" << polygon_filament.name() << "'"
<< '\n';
} else {
std::cout << prefix << "  name: none" << '\n';
}

if (polygon_filament.vertices_size() > 0) {
std::cout << prefix << "  vertices: [" << polygon_filament.vertices_size()
<< "]" << '\n';
} else {
std::cout << prefix << "  vertices: none" << '\n';
}

std::cout << prefix << "}" << '\n';
}

void PrintCurrentCarrier(const CurrentCarrier& current_carrier,
int indentation) {
std::string prefix;
for (int i = 0; i < indentation; ++i) {
prefix += " ";
}

std::cout << prefix << "CurrentCarrier {" << '\n';

switch (current_carrier.type_case()) {
case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
PrintInfiniteStraightFilament(
current_carrier.infinite_straight_filament(), indentation + 2);
break;
case CurrentCarrier::TypeCase::kCircularFilament:
PrintCircularFilament(current_carrier.circular_filament(),
indentation + 2);
break;
case CurrentCarrier::TypeCase::kPolygonFilament:
PrintPolygonFilament(current_carrier.polygon_filament(), indentation + 2);
break;
case CurrentCarrier::TypeCase::kTypeNotSet:

break;
default:
std::stringstream error_message;
error_message << "current carrier type ";
error_message << current_carrier.type_case();
error_message << " not implemented yet.";
LOG(FATAL) << error_message.str();
}

std::cout << prefix << "}" << '\n';
}

void PrintCoil(const Coil& coil, int indentation) {
std::string prefix;
for (int i = 0; i < indentation; ++i) {
prefix += " ";
}

std::cout << prefix << "Coil {" << '\n';

if (coil.has_name()) {
std::cout << prefix << "  name: '" << coil.name() << "'" << '\n';
} else {
std::cout << prefix << "  name: none" << '\n';
}

if (coil.has_num_windings()) {
std::cout << prefix << "  num_windings: " << coil.num_windings() << '\n';
} else {
std::cout << prefix << "  num_windings: none" << '\n';
}

for (const CurrentCarrier& current_carrier : coil.current_carriers()) {
PrintCurrentCarrier(current_carrier, indentation + 2);
}

std::cout << prefix << "}" << '\n';
}

void PrintSerialCircuit(const SerialCircuit& serial_circuit, int indentation) {
std::string prefix;
for (int i = 0; i < indentation; ++i) {
prefix += " ";
}

std::cout << prefix << "SerialCircuit {" << '\n';

if (serial_circuit.has_name()) {
std::cout << prefix << "  name: '" << serial_circuit.name() << "'" << '\n';
} else {
std::cout << prefix << "  name: none" << '\n';
}

if (serial_circuit.has_current()) {
std::cout << prefix << "  current: " << serial_circuit.current() << '\n';
} else {
std::cout << prefix << "  current: none" << '\n';
}

for (const Coil& coil : serial_circuit.coils()) {
PrintCoil(coil, indentation + 2);
}

std::cout << prefix << "}" << '\n';
}

void PrintMagneticConfiguration(
const MagneticConfiguration& magnetic_configuration, int indentation) {
std::string prefix;
for (int i = 0; i < indentation; ++i) {
prefix += " ";
}

std::cout << prefix << "MagneticConfiguration {" << '\n';

if (magnetic_configuration.has_name()) {
std::cout << prefix << "  name: '" << magnetic_configuration.name() << "'"
<< '\n';
} else {
std::cout << prefix << "  name: none" << '\n';
}

if (magnetic_configuration.has_num_field_periods()) {
std::cout << prefix << "  num_field_periods: "
<< magnetic_configuration.num_field_periods() << '\n';
} else {
std::cout << prefix << "  num_field_periods: none" << '\n';
}

for (const SerialCircuit& serial_circuit :
magnetic_configuration.serial_circuits()) {
PrintSerialCircuit(serial_circuit, indentation + 2);
}

std::cout << prefix << "}" << '\n';
}

}

// source: vmecpp/common/magnetic_field_provider/magnetic_field_provider_lib.cc

// header: vmecpp/common/magnetic_field_provider/magnetic_field_provider_lib.h

#ifndef VMECPP_COMMON_MAGNETIC_FIELD_PROVIDER_MAGNETIC_FIELD_PROVIDER_LIB_H_
#define VMECPP_COMMON_MAGNETIC_FIELD_PROVIDER_MAGNETIC_FIELD_PROVIDER_LIB_H_

#include <Eigen/Dense>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace magnetics {

using RowMatrix3Xd = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>;

absl::Status MagneticField(
const InfiniteStraightFilament &infinite_straight_filament, double current,
const std::vector<std::vector<double> > &evaluation_positions,
std::vector<std::vector<double> > &m_magnetic_field,
bool check_current_carrier = true);

absl::Status MagneticField(
const CircularFilament &circular_filament, double current,
const std::vector<std::vector<double> > &evaluation_positions,
std::vector<std::vector<double> > &m_magnetic_field,
bool check_current_carrier = true);

absl::Status MagneticField(
const PolygonFilament &polygon_filament, double current,
const std::vector<std::vector<double> > &evaluation_positions,
std::vector<std::vector<double> > &m_magnetic_field,
bool check_current_carrier = true);

absl::Status MagneticField(
const MagneticConfiguration &magnetic_configuration,
const std::vector<std::vector<double> > &evaluation_positions,
std::vector<std::vector<double> > &m_magnetic_field,
bool check_current_carrier = true);

absl::Status MagneticField(const MagneticConfiguration &magnetic_configuration,
const RowMatrix3Xd &evaluation_positions,
RowMatrix3Xd &m_magnetic_field,
bool check_current_carrier = true);

absl::Status VectorPotential(
const CircularFilament &circular_filament, double current,
const std::vector<std::vector<double> > &evaluation_positions,
std::vector<std::vector<double> > &m_vector_potential,
bool check_current_carrier = true);

absl::Status VectorPotential(
const PolygonFilament &polygon_filament, double current,
const std::vector<std::vector<double> > &evaluation_positions,
std::vector<std::vector<double> > &m_vector_potential,
bool check_current_carrier = true);

absl::Status VectorPotential(
const MagneticConfiguration &magnetic_configuration,
const std::vector<std::vector<double> > &evaluation_positions,
std::vector<std::vector<double> > &m_vector_potential,
bool check_current_carrier = true);

absl::Status VectorPotential(
const MagneticConfiguration &magnetic_configuration,
const RowMatrix3Xd &evaluation_positions, RowMatrix3Xd &m_vector_potential,
bool check_current_carrier = true);

absl::StatusOr<double> LinkingCurrent(
const MagneticConfiguration &magnetic_configuration,
const composed_types::CurveRZFourier &axis_coefficients);

}

#endif

#include <algorithm>
#include <cstddef>
#include <span>
#include <sstream>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"

namespace magnetics {

using composed_types::CurveRZFourier;
using composed_types::FourierCoefficient1D;
using composed_types::Normalize;
using composed_types::Vector3d;

namespace {

std::vector<double> ToAbscabOrder(
const std::vector<std::vector<double>>& positions) {
const std::size_t number_of_positions = positions.size();
std::vector<double> flat(number_of_positions * 3);
for (std::size_t i = 0; i < number_of_positions; ++i) {
flat[i * 3 + 0] = positions[i][0];
flat[i * 3 + 1] = positions[i][1];
flat[i * 3 + 2] = positions[i][2];
}
return flat;
}

std::vector<double> ToAbscabOrder(const RowMatrix3Xd& positions) {
const auto number_of_positions = static_cast<std::size_t>(positions.cols());
std::vector<double> flat(number_of_positions * 3);
for (std::size_t i = 0; i < number_of_positions; ++i) {
const auto column = static_cast<Eigen::Index>(i);
flat[i * 3 + 0] = positions(0, column);
flat[i * 3 + 1] = positions(1, column);
flat[i * 3 + 2] = positions(2, column);
}
return flat;
}

std::vector<double> VerticesInAbscabOrder(
const PolygonFilament& polygon_filament) {
const int number_of_vertices = polygon_filament.vertices_size();
std::vector<double> flat(static_cast<std::size_t>(number_of_vertices) * 3);
for (int i = 0; i < number_of_vertices; ++i) {
const Vector3d& vertex = polygon_filament.vertices(i);
flat[i * 3 + 0] = vertex.x();
flat[i * 3 + 1] = vertex.y();
flat[i * 3 + 2] = vertex.z();
}
return flat;
}

void AddFromAbscabOrder(const std::vector<double>& contribution,
std::vector<std::vector<double>>& m_target) {
for (std::size_t i = 0; i < m_target.size(); ++i) {
m_target[i][0] += contribution[i * 3 + 0];
m_target[i][1] += contribution[i * 3 + 1];
m_target[i][2] += contribution[i * 3 + 2];
}
}

void AddFromAbscabOrder(const std::vector<double>& contribution,
RowMatrix3Xd& m_target) {
for (Eigen::Index column = 0; column < m_target.cols(); ++column) {
const auto i = static_cast<std::size_t>(column);
m_target(0, column) += contribution[i * 3 + 0];
m_target(1, column) += contribution[i * 3 + 1];
m_target(2, column) += contribution[i * 3 + 2];
}
}

void AddMagneticField(
const InfiniteStraightFilament& infinite_straight_filament, double current,
std::span<double> evaluation_positions, std::span<double> m_contribution) {
if (current == 0.0) {

return;
}
const double magnetic_field_scale = abscab::MU_0 * current / (2.0 * M_PI);

const Vector3d& direction = infinite_straight_filament.direction();
const double direction_x = direction.x();
const double direction_y = direction.y();
const double direction_z = direction.z();
const double direction_length =
std::hypot(direction_x, direction_y, direction_z);

CHECK_GT(direction_length, 0.0);

const double normalized_direction_x = direction_x / direction_length;
const double normalized_direction_y = direction_y / direction_length;
const double normalized_direction_z = direction_z / direction_length;

const Vector3d& origin = infinite_straight_filament.origin();
const double origin_x = origin.x();
const double origin_y = origin.y();
const double origin_z = origin.z();

const std::size_t num_evaluation_locations = evaluation_positions.size() / 3;
for (std::size_t i = 0; i < num_evaluation_locations; ++i) {
const double evaluation_position_x = evaluation_positions[i * 3 + 0];
const double evaluation_position_y = evaluation_positions[i * 3 + 1];
const double evaluation_position_z = evaluation_positions[i * 3 + 2];

const double delta_eval_origin_x = origin_x - evaluation_position_x;
const double delta_eval_origin_y = origin_y - evaluation_position_y;
const double delta_eval_origin_z = origin_z - evaluation_position_z;

const double parallel_distance =
(delta_eval_origin_x * normalized_direction_x +
delta_eval_origin_y * normalized_direction_y +
delta_eval_origin_z * normalized_direction_z);

const double delta_parallel_x = normalized_direction_x * parallel_distance;
const double delta_parallel_y = normalized_direction_y * parallel_distance;
const double delta_parallel_z = normalized_direction_z * parallel_distance;

const double delta_perpendicular_x = delta_eval_origin_x - delta_parallel_x;
const double delta_perpendicular_y = delta_eval_origin_y - delta_parallel_y;
const double delta_perpendicular_z = delta_eval_origin_z - delta_parallel_z;

const double evaluation_position_radius = std::hypot(
delta_perpendicular_x, delta_perpendicular_y, delta_perpendicular_z);

CHECK_GT(evaluation_position_radius, 0.0);

const double magnetic_field_strength =
magnetic_field_scale / evaluation_position_radius;

const double radial_unit_vector_x =
delta_perpendicular_x / evaluation_position_radius;
const double radial_unit_vector_y =
delta_perpendicular_y / evaluation_position_radius;
const double radial_unit_vector_z =
delta_perpendicular_z / evaluation_position_radius;

const double toroidal_unit_vector_x =
radial_unit_vector_y * normalized_direction_z -
radial_unit_vector_z * normalized_direction_y;
const double toroidal_unit_vector_y =
radial_unit_vector_z * normalized_direction_x -
radial_unit_vector_x * normalized_direction_z;
const double toroidal_unit_vector_z =
radial_unit_vector_x * normalized_direction_y -
radial_unit_vector_y * normalized_direction_x;

const double magnetic_field_vector_x =
toroidal_unit_vector_x * magnetic_field_strength;
const double magnetic_field_vector_y =
toroidal_unit_vector_y * magnetic_field_strength;
const double magnetic_field_vector_z =
toroidal_unit_vector_z * magnetic_field_strength;

m_contribution[i * 3 + 0] += magnetic_field_vector_x;
m_contribution[i * 3 + 1] += magnetic_field_vector_y;
m_contribution[i * 3 + 2] += magnetic_field_vector_z;
}
}

void AddMagneticField(const CircularFilament& circular_filament, double current,
std::span<double> evaluation_positions,
std::span<double> m_contribution) {
const Vector3d& center_vector = circular_filament.center();
std::vector<double> center = {
center_vector.x(),
center_vector.y(),
center_vector.z(),
};

const Vector3d& normal_vector = circular_filament.normal();
std::vector<double> normal = {
normal_vector.x(),
normal_vector.y(),
normal_vector.z(),
};

const double radius = circular_filament.radius();

abscab::magneticFieldCircularFilament(
center.data(), normal.data(), radius, current,
static_cast<int>(evaluation_positions.size() / 3),
evaluation_positions.data(), m_contribution.data());
}

void AddMagneticField(const PolygonFilament& polygon_filament, double current,
std::span<double> evaluation_positions,
std::span<double> m_contribution) {
std::vector<double> vertices = VerticesInAbscabOrder(polygon_filament);

abscab::magneticFieldPolygonFilament(
polygon_filament.vertices_size(), vertices.data(), current,
static_cast<int>(evaluation_positions.size() / 3),
evaluation_positions.data(), m_contribution.data());
}

void AddVectorPotential(const CircularFilament& circular_filament,
double current, std::span<double> evaluation_positions,
std::span<double> m_contribution) {
const Vector3d& center_vector = circular_filament.center();
std::vector<double> center = {
center_vector.x(),
center_vector.y(),
center_vector.z(),
};

const Vector3d& normal_vector = circular_filament.normal();
std::vector<double> normal = {
normal_vector.x(),
normal_vector.y(),
normal_vector.z(),
};

const double radius = circular_filament.radius();

abscab::vectorPotentialCircularFilament(
center.data(), normal.data(), radius, -current,
static_cast<int>(evaluation_positions.size() / 3),
evaluation_positions.data(), m_contribution.data());
}

void AddVectorPotential(const PolygonFilament& polygon_filament, double current,
std::span<double> evaluation_positions,
std::span<double> m_contribution) {
std::vector<double> vertices = VerticesInAbscabOrder(polygon_filament);

abscab::vectorPotentialPolygonFilament(
polygon_filament.vertices_size(), vertices.data(), current,
static_cast<int>(evaluation_positions.size() / 3),
evaluation_positions.data(), m_contribution.data());
}

double CoilCurrent(const SerialCircuit& serial_circuit, const Coil& coil) {
if (coil.has_num_windings()) {
return serial_circuit.current() * coil.num_windings();
}

return serial_circuit.current();
}

absl::Status UnsupportedCurrentCarrier(const CurrentCarrier& current_carrier) {
std::stringstream error_message;
error_message << "current carrier type ";
error_message << current_carrier.type_case();
error_message << " not implemented yet.";
return absl::InvalidArgumentError(error_message.str());
}

template <typename Target>
absl::Status AccumulateMagneticField(
const MagneticConfiguration& magnetic_configuration,
std::span<double> evaluation_positions, Target& m_target) {
std::vector<double> contribution(evaluation_positions.size(), 0.0);

for (const SerialCircuit& serial_circuit :
magnetic_configuration.serial_circuits()) {
if (!serial_circuit.has_current() || serial_circuit.current() == 0.0) {

continue;
}

for (const Coil& coil : serial_circuit.coils()) {
const double current = CoilCurrent(serial_circuit, coil);

for (const CurrentCarrier& current_carrier : coil.current_carriers()) {
std::fill(contribution.begin(), contribution.end(), 0.0);
switch (current_carrier.type_case()) {
case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
AddMagneticField(current_carrier.infinite_straight_filament(),
current, evaluation_positions, contribution);
break;
case CurrentCarrier::TypeCase::kCircularFilament:
AddMagneticField(current_carrier.circular_filament(), current,
evaluation_positions, contribution);
break;
case CurrentCarrier::TypeCase::kPolygonFilament:
AddMagneticField(current_carrier.polygon_filament(), current,
evaluation_positions, contribution);
break;
case CurrentCarrier::TypeCase::kTypeNotSet:

continue;
default:
return UnsupportedCurrentCarrier(current_carrier);
}
AddFromAbscabOrder(contribution, m_target);
}
}
}

return absl::OkStatus();
}

template <typename Target>
absl::Status AccumulateVectorPotential(
const MagneticConfiguration& magnetic_configuration,
std::span<double> evaluation_positions, Target& m_target) {
std::vector<double> contribution(evaluation_positions.size(), 0.0);

for (const SerialCircuit& serial_circuit :
magnetic_configuration.serial_circuits()) {
if (!serial_circuit.has_current() || serial_circuit.current() == 0.0) {

continue;
}

for (const Coil& coil : serial_circuit.coils()) {
const double current = CoilCurrent(serial_circuit, coil);

for (const CurrentCarrier& current_carrier : coil.current_carriers()) {
std::fill(contribution.begin(), contribution.end(), 0.0);
switch (current_carrier.type_case()) {
case CurrentCarrier::TypeCase::kInfiniteStraightFilament:

LOG(FATAL) << "Cannot compute the magnetic vector potential of an "
"infinite straight filament.";
break;
case CurrentCarrier::TypeCase::kCircularFilament:
AddVectorPotential(current_carrier.circular_filament(), current,
evaluation_positions, contribution);
break;
case CurrentCarrier::TypeCase::kPolygonFilament:
AddVectorPotential(current_carrier.polygon_filament(), current,
evaluation_positions, contribution);
break;
case CurrentCarrier::TypeCase::kTypeNotSet:

continue;
default:
return UnsupportedCurrentCarrier(current_carrier);
}
AddFromAbscabOrder(contribution, m_target);
}
}
}

return absl::OkStatus();
}

absl::Status CheckFreeOfInfiniteStraightFilaments(
const MagneticConfiguration& magnetic_configuration) {
for (const SerialCircuit& serial_circuit :
magnetic_configuration.serial_circuits()) {
for (const Coil& coil : serial_circuit.coils()) {
for (const CurrentCarrier& current_carrier : coil.current_carriers()) {
if (current_carrier.has_infinite_straight_filament()) {
return absl::InvalidArgumentError(
"Cannot compute the magnetic vector potential of an infinite "
"straight filament.");
}
}
}
}
return absl::OkStatus();
}

}

absl::Status MagneticField(
const InfiniteStraightFilament& infinite_straight_filament, double current,
const std::vector<std::vector<double>>& evaluation_positions,
std::vector<std::vector<double>>& m_magnetic_field,
bool check_current_carrier) {
if (check_current_carrier) {
absl::Status status =
IsInfiniteStraightFilamentFullyPopulated(infinite_straight_filament);
if (!status.ok()) {

return status;
}
}

std::vector<double> evaluation_positions_flat =
ToAbscabOrder(evaluation_positions);
std::vector<double> contribution(evaluation_positions_flat.size(), 0.0);

AddMagneticField(infinite_straight_filament, current,
evaluation_positions_flat, contribution);
AddFromAbscabOrder(contribution, m_magnetic_field);

return absl::OkStatus();
}

absl::Status MagneticField(
const CircularFilament& circular_filament, double current,
const std::vector<std::vector<double>>& evaluation_positions,
std::vector<std::vector<double>>& m_magnetic_field,
bool check_current_carrier) {
if (check_current_carrier) {
absl::Status status = IsCircularFilamentFullyPopulated(circular_filament);
if (!status.ok()) {

return status;
}
}

std::vector<double> evaluation_positions_flat =
ToAbscabOrder(evaluation_positions);
std::vector<double> contribution(evaluation_positions_flat.size(), 0.0);

AddMagneticField(circular_filament, current, evaluation_positions_flat,
contribution);
AddFromAbscabOrder(contribution, m_magnetic_field);

return absl::OkStatus();
}

absl::Status MagneticField(
const PolygonFilament& polygon_filament, double current,
const std::vector<std::vector<double>>& evaluation_positions,
std::vector<std::vector<double>>& m_magnetic_field,
bool check_current_carrier) {
if (check_current_carrier) {
absl::Status status = IsPolygonFilamentFullyPopulated(polygon_filament);
if (!status.ok()) {

return status;
}
}

std::vector<double> evaluation_positions_flat =
ToAbscabOrder(evaluation_positions);
std::vector<double> contribution(evaluation_positions_flat.size(), 0.0);

AddMagneticField(polygon_filament, current, evaluation_positions_flat,
contribution);
AddFromAbscabOrder(contribution, m_magnetic_field);

return absl::OkStatus();
}

absl::Status MagneticField(
const MagneticConfiguration& magnetic_configuration,
const std::vector<std::vector<double>>& evaluation_positions,
std::vector<std::vector<double>>& m_magnetic_field,
bool check_current_carrier) {
if (check_current_carrier) {
absl::Status status =
IsMagneticConfigurationFullyPopulated(magnetic_configuration);
if (!status.ok()) {

return status;
}
}

std::vector<double> evaluation_positions_flat =
ToAbscabOrder(evaluation_positions);
return AccumulateMagneticField(magnetic_configuration,
evaluation_positions_flat, m_magnetic_field);
}

absl::Status MagneticField(const MagneticConfiguration& magnetic_configuration,
const RowMatrix3Xd& evaluation_positions,
RowMatrix3Xd& m_magnetic_field,
bool check_current_carrier) {
CHECK_EQ(m_magnetic_field.cols(), evaluation_positions.cols())
<< "one magnetic field vector per evaluation position is required";

if (check_current_carrier) {
absl::Status status =
IsMagneticConfigurationFullyPopulated(magnetic_configuration);
if (!status.ok()) {

return status;
}
}

std::vector<double> evaluation_positions_flat =
ToAbscabOrder(evaluation_positions);
return AccumulateMagneticField(magnetic_configuration,
evaluation_positions_flat, m_magnetic_field);
}

absl::Status VectorPotential(
const CircularFilament& circular_filament, double current,
const std::vector<std::vector<double>>& evaluation_positions,
std::vector<std::vector<double>>& m_vector_potential,
bool check_current_carrier) {
if (check_current_carrier) {
absl::Status status = IsCircularFilamentFullyPopulated(circular_filament);
if (!status.ok()) {

return status;
}
}

std::vector<double> evaluation_positions_flat =
ToAbscabOrder(evaluation_positions);
std::vector<double> contribution(evaluation_positions_flat.size(), 0.0);

AddVectorPotential(circular_filament, current, evaluation_positions_flat,
contribution);
AddFromAbscabOrder(contribution, m_vector_potential);

return absl::OkStatus();
}

absl::Status VectorPotential(
const PolygonFilament& polygon_filament, double current,
const std::vector<std::vector<double>>& evaluation_positions,
std::vector<std::vector<double>>& m_vector_potential,
bool check_current_carrier) {
if (check_current_carrier) {
absl::Status status = IsPolygonFilamentFullyPopulated(polygon_filament);
if (!status.ok()) {

return status;
}
}

std::vector<double> evaluation_positions_flat =
ToAbscabOrder(evaluation_positions);
std::vector<double> contribution(evaluation_positions_flat.size(), 0.0);

AddVectorPotential(polygon_filament, current, evaluation_positions_flat,
contribution);
AddFromAbscabOrder(contribution, m_vector_potential);

return absl::OkStatus();
}

absl::Status VectorPotential(
const MagneticConfiguration& magnetic_configuration,
const std::vector<std::vector<double>>& evaluation_positions,
std::vector<std::vector<double>>& m_vector_potential,
bool check_current_carrier) {
if (check_current_carrier) {
absl::Status status =
IsMagneticConfigurationFullyPopulated(magnetic_configuration);
if (!status.ok()) {

return status;
}

status = CheckFreeOfInfiniteStraightFilaments(magnetic_configuration);
if (!status.ok()) {
return status;
}
}

std::vector<double> evaluation_positions_flat =
ToAbscabOrder(evaluation_positions);
return AccumulateVectorPotential(
magnetic_configuration, evaluation_positions_flat, m_vector_potential);
}

absl::Status VectorPotential(
const MagneticConfiguration& magnetic_configuration,
const RowMatrix3Xd& evaluation_positions, RowMatrix3Xd& m_vector_potential,
bool check_current_carrier) {
CHECK_EQ(m_vector_potential.cols(), evaluation_positions.cols())
<< "one vector potential per evaluation position is required";

if (check_current_carrier) {
absl::Status status =
IsMagneticConfigurationFullyPopulated(magnetic_configuration);
if (!status.ok()) {

return status;
}

status = CheckFreeOfInfiniteStraightFilaments(magnetic_configuration);
if (!status.ok()) {
return status;
}
}

std::vector<double> evaluation_positions_flat =
ToAbscabOrder(evaluation_positions);
return AccumulateVectorPotential(
magnetic_configuration, evaluation_positions_flat, m_vector_potential);
}

absl::StatusOr<double> LinkingCurrent(
const MagneticConfiguration& magnetic_configuration,
const CurveRZFourier& axis_coefficients) {
static constexpr double kMu0 = 4.0e-7 * M_PI;

absl::Status status = IsCurveRZFourierFullyPopulated(axis_coefficients);
if (!status.ok()) {
return status;
}
const int num_coefficients = axis_coefficients.r_size();

int maximum_mode_number = 0;
for (int coefficient_index = 0; coefficient_index < num_coefficients;
++coefficient_index) {
maximum_mode_number =
std::max(maximum_mode_number,
axis_coefficients.r(coefficient_index).mode_number());
maximum_mode_number =
std::max(maximum_mode_number,
axis_coefficients.z(coefficient_index).mode_number());
}

const int num_axis_points = 2 * (2 * maximum_mode_number + 1) *
magnetic_configuration.num_field_periods();

std::vector<std::vector<double>> axis_points(num_axis_points);
std::vector<std::vector<double>> axis_tangent(num_axis_points);
const double delta_phi = 2.0 * M_PI / num_axis_points;
for (int k = 0; k < num_axis_points; ++k) {
const double cos_phi = std::cos(k * delta_phi);
const double sin_phi = std::sin(k * delta_phi);

double axis_point_r = 0.0;
double axis_point_z = 0.0;
double axis_tangent_r = 0.0;
double axis_tangent_z = 0.0;
for (int coefficient_index = 0; coefficient_index < num_coefficients;
++coefficient_index) {
const FourierCoefficient1D& coeff_r =
axis_coefficients.r(coefficient_index);
const FourierCoefficient1D& coeff_z =
axis_coefficients.z(coefficient_index);

const int mode_number = coeff_r.mode_number();

const double kernel = k * mode_number * delta_phi;
const double cos_kernel = std::cos(kernel);
const double sin_kernel = std::sin(kernel);

if (coeff_r.has_fc_cos()) {
const double coeff = coeff_r.fc_cos();
axis_point_r += coeff * cos_kernel;
axis_tangent_r += coeff * mode_number * (-sin_kernel);
}
if (coeff_r.has_fc_sin()) {
const double coeff = coeff_r.fc_sin();
axis_point_r += coeff * sin_kernel;
axis_tangent_r += coeff * mode_number * cos_kernel;
}

if (coeff_z.has_fc_cos()) {
const double coeff = coeff_z.fc_cos();
axis_point_z += coeff * cos_kernel;
axis_tangent_z += coeff * mode_number * (-sin_kernel);
}
if (coeff_z.has_fc_sin()) {
const double coeff = coeff_z.fc_sin();
axis_point_z += coeff * sin_kernel;
axis_tangent_z += coeff * mode_number * cos_kernel;
}
}

const double axis_point_x = axis_point_r * cos_phi;
const double axis_point_y = axis_point_r * sin_phi;
axis_points[k] = {axis_point_x, axis_point_y, axis_point_z};

const double axis_tangent_x =
axis_tangent_r * cos_phi - axis_point_r * sin_phi;
const double axis_tangent_y =
axis_tangent_r * sin_phi + axis_point_r * cos_phi;
axis_tangent[k] = {axis_tangent_x, axis_tangent_y, axis_tangent_z};
}

std::vector<std::vector<double>> magnetic_field(num_axis_points,
std::vector<double>(3));
status = MagneticField(magnetic_configuration, axis_points,
magnetic_field);
if (!status.ok()) {
return status;
}

double linking_current = 0.0;
for (int k = 0; k < num_axis_points; ++k) {
const double b_dot_tangent = magnetic_field[k][0] * axis_tangent[k][0] +
magnetic_field[k][1] * axis_tangent[k][1] +
magnetic_field[k][2] * axis_tangent[k][2];
linking_current += b_dot_tangent;
}

linking_current *= 2.0 * M_PI / num_axis_points;

linking_current /= kMu0;

return linking_current;
}

}

// source: vmecpp/common/makegrid_lib/makegrid_lib.cc

// header: vmecpp/common/makegrid_lib/makegrid_lib.h

#ifndef VMECPP_COMMON_MAKEGRID_LIB_MAKEGRID_LIB_H_
#define VMECPP_COMMON_MAKEGRID_LIB_MAKEGRID_LIB_H_

#include <Eigen/Dense>
#include <filesystem>
#include <string>

#include "absl/status/statusor.h"

namespace makegrid {

using magnetics::MagneticConfiguration;

using vmecpp::RowMatrixXd;
using RowMatrix3Xd = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>;

struct MakegridParameters {

bool normalize_by_currents = false;

bool assume_stellarator_symmetry = false;

int number_of_field_periods = 0;

double r_grid_minimum = 0.0;

double r_grid_maximum = 0.0;

int number_of_r_grid_points = 0;

double z_grid_minimum = 0.0;

double z_grid_maximum = 0.0;

int number_of_z_grid_points = 0;

int number_of_phi_grid_points = 0;
};

struct MagneticFieldResponseTable {

MakegridParameters parameters;

RowMatrixXd b_r;

RowMatrixXd b_p;

RowMatrixXd b_z;
};

struct MakegridCachedVectorPotential {

MakegridParameters parameters;

RowMatrixXd a_r;

RowMatrixXd a_p;

RowMatrixXd a_z;
};

absl::Status IsValidMakegridParameters(
const MakegridParameters& makegrid_parameters);

absl::StatusOr<MakegridParameters> ImportMakegridParametersFromJson(
const std::string& makegrid_parameters_json);

absl::StatusOr<MakegridParameters> ImportMakegridParametersFromFile(
const std::filesystem::path& makegrid_parameters_file);

absl::StatusOr<RowMatrix3Xd> MakeCylindricalGrid(
const MakegridParameters& makegrid_parameters);

absl::StatusOr<MagneticFieldResponseTable> ComputeMagneticFieldResponseTable(
const MakegridParameters& makegrid_parameters,
const MagneticConfiguration& magnetic_configuration);

absl::StatusOr<MakegridCachedVectorPotential> ComputeVectorPotentialCache(
const MakegridParameters& makegrid_parameters,
const MagneticConfiguration& magnetic_configuration);

absl::Status WriteMakegridNetCDFFile(
const std::string& makegrid_filename,
const MakegridParameters& makegrid_parameters,
const Eigen::VectorXd& circuit_currents,
const MagneticFieldResponseTable& magnetic_response_table,
const std::optional<MakegridCachedVectorPotential>& vector_potential_cache);

}

#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "netcdf.h"
#include "nlohmann/json.hpp"
namespace makegrid {

using nlohmann::json;

using json_io::JsonParse;
using json_io::JsonReadBool;
using json_io::JsonReadDouble;
using json_io::JsonReadInt;

using magnetics::GetCircuitCurrents;
using magnetics::MagneticField;
using magnetics::NumWindingsToCircuitCurrents;
using magnetics::SetCircuitCurrents;
using magnetics::VectorPotential;

absl::Status IsValidMakegridParameters(
const MakegridParameters& makegrid_parameters) {

if (makegrid_parameters.number_of_field_periods <= 0) {
return absl::InvalidArgumentError(
absl::StrFormat("number_of_field_periods must be > 0, but is %d",
makegrid_parameters.number_of_field_periods));
}

if (makegrid_parameters.r_grid_maximum <=
makegrid_parameters.r_grid_minimum) {
return absl::InvalidArgumentError(
absl::StrFormat("R grid extent must be positive, but is from "
"r_grid_minimum = % .3e to r_grid_maximum = % .3e",
makegrid_parameters.r_grid_minimum,
makegrid_parameters.r_grid_maximum));
}

if (makegrid_parameters.number_of_r_grid_points < 2) {
return absl::InvalidArgumentError(
absl::StrFormat("number_of_r_grid_points must be > 1, but is %d",
makegrid_parameters.number_of_r_grid_points));
}

if (makegrid_parameters.z_grid_maximum <=
makegrid_parameters.z_grid_minimum) {
return absl::InvalidArgumentError(
absl::StrFormat("Z grid extent must be positive, but is from "
"z_grid_minimum = % .3e to z_grid_maximum = % .3e",
makegrid_parameters.z_grid_minimum,
makegrid_parameters.z_grid_maximum));
}

if (makegrid_parameters.number_of_z_grid_points < 2) {
return absl::InvalidArgumentError(
absl::StrFormat("number_of_z_grid_points must be > 1, but is %d",
makegrid_parameters.number_of_z_grid_points));
}

constexpr double kZGridSymmetryTolerance =
64.0 * std::numeric_limits<double>::epsilon();
const double z_grid_symmetry_error =
makegrid_parameters.z_grid_minimum + makegrid_parameters.z_grid_maximum;
const double z_grid_symmetry_scale =
1.0 + std::max(std::abs(makegrid_parameters.z_grid_minimum),
std::abs(makegrid_parameters.z_grid_maximum));
if (makegrid_parameters.assume_stellarator_symmetry &&
std::abs(z_grid_symmetry_error) >
kZGridSymmetryTolerance * z_grid_symmetry_scale) {
return absl::InvalidArgumentError(absl::StrFormat(
"assume_stellarator_symmetry needs z_grid_minimum = -z_grid_maximum, "
"but the Z grid runs from z_grid_minimum = % .3e to z_grid_maximum = "
"% .3e, which sum to % .3e",
makegrid_parameters.z_grid_minimum, makegrid_parameters.z_grid_maximum,
z_grid_symmetry_error));
}

if (makegrid_parameters.number_of_phi_grid_points < 1) {
return absl::InvalidArgumentError(
absl::StrFormat("number_of_phi_grid_points must be > 0, but is %d",
makegrid_parameters.number_of_phi_grid_points));
}

return absl::OkStatus();
}

absl::StatusOr<MakegridParameters> ImportMakegridParametersFromJson(
const std::string& makegrid_parameters_json) {
absl::StatusOr<json> maybe_json = JsonParse(makegrid_parameters_json);
if (!maybe_json.ok()) {
return maybe_json.status();
}
const json& j = *maybe_json;

MakegridParameters makegrid_parameters;

absl::StatusOr<std::optional<bool>> maybe_normalize_by_currents =
JsonReadBool(j, "normalize_by_currents");
if (!maybe_normalize_by_currents.ok()) {
return maybe_normalize_by_currents.status();
} else {
if (maybe_normalize_by_currents->has_value()) {
makegrid_parameters.normalize_by_currents =
maybe_normalize_by_currents->value();
} else {

return absl::NotFoundError(
"'normalize_by_currents':bool must be specified in "
"MakegridParameters");
}
}

absl::StatusOr<std::optional<bool>> maybe_assume_stellarator_symmetry =
JsonReadBool(j, "assume_stellarator_symmetry");
if (!maybe_assume_stellarator_symmetry.ok()) {
return maybe_assume_stellarator_symmetry.status();
} else {
if (maybe_assume_stellarator_symmetry->has_value()) {
makegrid_parameters.assume_stellarator_symmetry =
maybe_assume_stellarator_symmetry->value();
} else {

return absl::NotFoundError(
"'assume_stellarator_symmetry':bool must be specified in "
"MakegridParameters");
}
}

absl::StatusOr<std::optional<int>> maybe_number_of_field_periods =
JsonReadInt(j, "number_of_field_periods");
if (!maybe_number_of_field_periods.ok()) {
return maybe_number_of_field_periods.status();
} else {
if (maybe_number_of_field_periods->has_value()) {
makegrid_parameters.number_of_field_periods =
maybe_number_of_field_periods->value();
} else {

return absl::NotFoundError(
"'number_of_field_periods':int must be specified in "
"MakegridParameters");
}
}

absl::StatusOr<std::optional<double>> maybe_r_grid_minimum =
JsonReadDouble(j, "r_grid_minimum");
if (!maybe_r_grid_minimum.ok()) {
return maybe_r_grid_minimum.status();
} else {
if (maybe_r_grid_minimum->has_value()) {
makegrid_parameters.r_grid_minimum = maybe_r_grid_minimum->value();
} else {

return absl::NotFoundError(
"'r_grid_minimum':double must be specified in MakegridParameters");
}
}

absl::StatusOr<std::optional<double>> maybe_r_grid_maximum =
JsonReadDouble(j, "r_grid_maximum");
if (!maybe_r_grid_maximum.ok()) {
return maybe_r_grid_maximum.status();
} else {
if (maybe_r_grid_maximum->has_value()) {
makegrid_parameters.r_grid_maximum = maybe_r_grid_maximum->value();
} else {

return absl::NotFoundError(
"'r_grid_maximum':double must be specified in MakegridParameters");
}
}

absl::StatusOr<std::optional<int>> maybe_number_of_r_grid_points =
JsonReadInt(j, "number_of_r_grid_points");
if (!maybe_number_of_r_grid_points.ok()) {
return maybe_number_of_r_grid_points.status();
} else {
if (maybe_number_of_r_grid_points->has_value()) {
makegrid_parameters.number_of_r_grid_points =
maybe_number_of_r_grid_points->value();
} else {

return absl::NotFoundError(
"'number_of_r_grid_points':int must be specified in "
"MakegridParameters");
}
}

absl::StatusOr<std::optional<double>> maybe_z_grid_minimum =
JsonReadDouble(j, "z_grid_minimum");
if (!maybe_z_grid_minimum.ok()) {
return maybe_z_grid_minimum.status();
} else {
if (maybe_z_grid_minimum->has_value()) {
makegrid_parameters.z_grid_minimum = maybe_z_grid_minimum->value();
} else {

return absl::NotFoundError(
"'z_grid_minimum':double must be specified in MakegridParameters");
}
}

absl::StatusOr<std::optional<double>> maybe_z_grid_maximum =
JsonReadDouble(j, "z_grid_maximum");
if (!maybe_z_grid_maximum.ok()) {
return maybe_z_grid_maximum.status();
} else {
if (maybe_z_grid_maximum->has_value()) {
makegrid_parameters.z_grid_maximum = maybe_z_grid_maximum->value();
} else {

return absl::NotFoundError(
"'z_grid_maximum':double must be specified in MakegridParameters");
}
}

absl::StatusOr<std::optional<int>> maybe_number_of_z_grid_points =
JsonReadInt(j, "number_of_z_grid_points");
if (!maybe_number_of_z_grid_points.ok()) {
return maybe_number_of_z_grid_points.status();
} else {
if (maybe_number_of_z_grid_points->has_value()) {
makegrid_parameters.number_of_z_grid_points =
maybe_number_of_z_grid_points->value();
} else {

return absl::NotFoundError(
"'number_of_z_grid_points':int must be specified in "
"MakegridParameters");
}
}

absl::StatusOr<std::optional<int>> maybe_number_of_phi_grid_points =
JsonReadInt(j, "number_of_phi_grid_points");
if (!maybe_number_of_phi_grid_points.ok()) {
return maybe_number_of_phi_grid_points.status();
} else {
if (maybe_number_of_phi_grid_points->has_value()) {
makegrid_parameters.number_of_phi_grid_points =
maybe_number_of_phi_grid_points->value();
} else {

return absl::NotFoundError(
"'number_of_phi_grid_points':int must be specified in "
"MakegridParameters");
}
}

absl::Status makegrid_parameters_status =
IsValidMakegridParameters(makegrid_parameters);
if (!makegrid_parameters_status.ok()) {
return makegrid_parameters_status;
}

return makegrid_parameters;
}

absl::StatusOr<MakegridParameters> ImportMakegridParametersFromFile(
const std::filesystem::path& makegrid_parameters_file) {
const auto maybe_makegrid_params_json =
file_io::ReadFile(makegrid_parameters_file);
if (!maybe_makegrid_params_json.ok()) {
return maybe_makegrid_params_json.status();
}
const auto& makegrid_params_json = *maybe_makegrid_params_json;
return ImportMakegridParametersFromJson(makegrid_params_json);
}

absl::StatusOr<RowMatrix3Xd> MakeCylindricalGrid(
const MakegridParameters& makegrid_parameters) {
absl::Status makegrid_parameters_status =
IsValidMakegridParameters(makegrid_parameters);
if (!makegrid_parameters_status.ok()) {
return makegrid_parameters_status;
}

const int num_field_periods = makegrid_parameters.number_of_field_periods;
const int num_phi = makegrid_parameters.number_of_phi_grid_points;
const int num_z = makegrid_parameters.number_of_z_grid_points;
const int num_r = makegrid_parameters.number_of_r_grid_points;

const double min_r = makegrid_parameters.r_grid_minimum;
const double max_r = makegrid_parameters.r_grid_maximum;
const double min_z = makegrid_parameters.z_grid_minimum;
const double max_z = makegrid_parameters.z_grid_maximum;

const double delta_r = (max_r - min_r) / (num_r - 1.0);
const double delta_z = (max_z - min_z) / (num_z - 1.0);
const double delta_phi = 2.0 * M_PI / (num_field_periods * num_phi);

int num_phi_effective = num_phi;
if (makegrid_parameters.assume_stellarator_symmetry) {

num_phi_effective = num_phi / 2 + 1;
}

const int total_number_of_grid_points = num_phi_effective * num_z * num_r;

RowMatrix3Xd cylindrical_grid{
RowMatrix3Xd::Zero(3, total_number_of_grid_points)};

for (int phi_index = 0; phi_index < num_phi_effective; ++phi_index) {
const double phi = phi_index * delta_phi;
const double cos_phi = std::cos(phi);
const double sin_phi = std::sin(phi);
for (int z_index = 0; z_index < num_z; ++z_index) {
const double z = min_z + z_index * delta_z;
for (int r_index = 0; r_index < num_r; ++r_index) {
const double r = min_r + r_index * delta_r;

const double x = r * cos_phi;
const double y = r * sin_phi;

const int linear_index =
(phi_index * num_z + z_index) * num_r + r_index;

cylindrical_grid(0, linear_index) = x;
cylindrical_grid(1, linear_index) = y;
cylindrical_grid(2, linear_index) = z;
}
}
}

return cylindrical_grid;
}

template <typename Derived3>
void CartesianToCylindricalField(
const Eigen::VectorXd& cos_phi, const Eigen::VectorXd& sin_phi,
const Eigen::DenseBase<Derived3>& magnetic_field, int num_z, int num_r,
int number_of_evaluation_points, Eigen::Ref<Eigen::RowVectorXd> m_b_r,
Eigen::Ref<Eigen::RowVectorXd> m_b_p,
Eigen::Ref<Eigen::RowVectorXd> m_b_z) {
const Eigen::Index num_phi = cos_phi.size();
const Eigen::Index total_number_of_grid_points = m_b_r.size();

CHECK_EQ(cos_phi.size(), sin_phi.size());
CHECK_EQ(m_b_p.size(), m_b_r.size());
CHECK_EQ(m_b_z.size(), m_b_r.size());
CHECK_EQ(num_phi * num_r * num_z, total_number_of_grid_points);

CHECK_LE(number_of_evaluation_points, total_number_of_grid_points);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (Eigen::Index linear_index = 0;
linear_index < number_of_evaluation_points; ++linear_index) {
const double b_x = magnetic_field(0, linear_index);
const double b_y = magnetic_field(1, linear_index);

const Eigen::Index index_phi = linear_index / (num_z * num_r);
const double b_r = b_x * cos_phi[index_phi] + b_y * sin_phi[index_phi];
const double b_p = b_y * cos_phi[index_phi] - b_x * sin_phi[index_phi];

m_b_r[linear_index] = b_r;
m_b_p[linear_index] = b_p;
}

m_b_z.head(number_of_evaluation_points) = magnetic_field.row(2);

if (number_of_evaluation_points < total_number_of_grid_points) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int linear_index = number_of_evaluation_points;
linear_index < total_number_of_grid_points; ++linear_index) {
const int idx_phi = linear_index / (num_z * num_r);
const int idx_z_r = linear_index % (num_z * num_r);
const int idx_z = idx_z_r / num_r;
const int idx_r = idx_z_r % num_r;

const int idx_phi_reversed = num_phi - idx_phi;
const int idx_z_reversed = num_z - 1 - idx_z;

const int linear_index_reversed =
(idx_phi_reversed * num_z + idx_z_reversed) * num_r + idx_r;

m_b_r[linear_index] = -m_b_r[linear_index_reversed];
m_b_p[linear_index] = m_b_p[linear_index_reversed];
m_b_z[linear_index] = m_b_z[linear_index_reversed];
}
}
}

absl::StatusOr<MagneticFieldResponseTable> ComputeMagneticFieldResponseTable(
const MakegridParameters& makegrid_parameters,
const MagneticConfiguration& magnetic_configuration) {
absl::StatusOr<RowMatrix3Xd> maybe_cylindrical_grid =
MakeCylindricalGrid(makegrid_parameters);
if (!maybe_cylindrical_grid.ok()) {
return maybe_cylindrical_grid.status();
}

const int number_of_evaluation_points =
static_cast<int>(maybe_cylindrical_grid.value().cols());

const int num_field_periods = makegrid_parameters.number_of_field_periods;
const int num_phi = makegrid_parameters.number_of_phi_grid_points;
const int num_z = makegrid_parameters.number_of_z_grid_points;
const int num_r = makegrid_parameters.number_of_r_grid_points;

const int total_number_of_grid_points = num_phi * num_z * num_r;

Eigen::VectorXd cos_phi{Eigen::VectorXd::Zero(num_phi)};
Eigen::VectorXd sin_phi{Eigen::VectorXd::Zero(num_phi)};
const double delta_phi = 2.0 * M_PI / (num_field_periods * num_phi);
for (int index_phi = 0; index_phi < num_phi; ++index_phi) {
const double phi = index_phi * delta_phi;
cos_phi(index_phi) = std::cos(phi);
sin_phi(index_phi) = std::sin(phi);
}

std::optional<MagneticConfiguration> migrated_configuration;
if (makegrid_parameters.normalize_by_currents) {
migrated_configuration = magnetic_configuration;
absl::Status migrate_status =
NumWindingsToCircuitCurrents(*migrated_configuration);
if (!migrate_status.ok()) {
return migrate_status;
}
}
const MagneticConfiguration& effective_configuration =
migrated_configuration.has_value() ? *migrated_configuration
: magnetic_configuration;

absl::StatusOr<Eigen::VectorXd> maybe_original_currents =
GetCircuitCurrents(effective_configuration);
if (!maybe_original_currents.ok()) {
return maybe_original_currents.status();
}
const Eigen::VectorXd& original_currents = *maybe_original_currents;

const int number_of_serial_circuits =
effective_configuration.serial_circuits_size();

MagneticFieldResponseTable response_table_b;
response_table_b.parameters = makegrid_parameters;

response_table_b.b_r.resize(number_of_serial_circuits,
total_number_of_grid_points);
response_table_b.b_p.resize(number_of_serial_circuits,
total_number_of_grid_points);
response_table_b.b_z.resize(number_of_serial_circuits,
total_number_of_grid_points);

std::vector<absl::Status> status(number_of_serial_circuits);

for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
++circuit_index) {

MagneticConfiguration m_magnetic_configuration = effective_configuration;

Eigen::VectorXd currents_for_circuit;
currents_for_circuit.setZero(number_of_serial_circuits);
if (makegrid_parameters.normalize_by_currents) {
currents_for_circuit[circuit_index] = 1.0;
} else {
currents_for_circuit[circuit_index] = original_currents[circuit_index];
}

absl::Status set_currents_status =
SetCircuitCurrents(currents_for_circuit, m_magnetic_configuration);
if (!set_currents_status.ok()) {
status[circuit_index] = set_currents_status;
continue;
}

RowMatrix3Xd magnetic_field =
RowMatrix3Xd::Zero(3, number_of_evaluation_points);

absl::Status magnetic_field_status =
MagneticField(m_magnetic_configuration, maybe_cylindrical_grid.value(),
magnetic_field);
if (!magnetic_field_status.ok()) {
status[circuit_index] = magnetic_field_status;
continue;
}
CartesianToCylindricalField(cos_phi, sin_phi, magnetic_field, num_z, num_r,
number_of_evaluation_points,
response_table_b.b_r.row(circuit_index),
response_table_b.b_p.row(circuit_index),
response_table_b.b_z.row(circuit_index));

LOG(INFO) << absl::StrFormat("B %2d/%2d: done", circuit_index + 1,
number_of_serial_circuits);
}

for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
++circuit_index) {
if (!status[circuit_index].ok()) {
return status[circuit_index];
}
}

return response_table_b;
}

absl::StatusOr<MakegridCachedVectorPotential> ComputeVectorPotentialCache(
const MakegridParameters& makegrid_parameters,
const MagneticConfiguration& magnetic_configuration) {
absl::StatusOr<RowMatrix3Xd> maybe_cylindrical_grid =
MakeCylindricalGrid(makegrid_parameters);
if (!maybe_cylindrical_grid.ok()) {
return maybe_cylindrical_grid.status();
}

const int number_of_evaluation_points =
static_cast<int>(maybe_cylindrical_grid.value().cols());

const int num_field_periods = makegrid_parameters.number_of_field_periods;
const int num_phi = makegrid_parameters.number_of_phi_grid_points;
const int num_z = makegrid_parameters.number_of_z_grid_points;
const int num_r = makegrid_parameters.number_of_r_grid_points;

const int total_number_of_grid_points = num_phi * num_z * num_r;

std::vector<double> cos_phi(num_phi);
std::vector<double> sin_phi(num_phi);
const double delta_phi = 2.0 * M_PI / (num_field_periods * num_phi);
for (int index_phi = 0; index_phi < num_phi; ++index_phi) {
const double phi = index_phi * delta_phi;
cos_phi[index_phi] = std::cos(phi);
sin_phi[index_phi] = std::sin(phi);
}

std::optional<MagneticConfiguration> migrated_configuration;
if (makegrid_parameters.normalize_by_currents) {
migrated_configuration = magnetic_configuration;
absl::Status migrate_status =
NumWindingsToCircuitCurrents(*migrated_configuration);
if (!migrate_status.ok()) {
return migrate_status;
}
}
const MagneticConfiguration& effective_configuration =
migrated_configuration.has_value() ? *migrated_configuration
: magnetic_configuration;

absl::StatusOr<Eigen::VectorXd> maybe_original_currents =
GetCircuitCurrents(effective_configuration);
if (!maybe_original_currents.ok()) {
return maybe_original_currents.status();
}
const Eigen::VectorXd& original_currents = *maybe_original_currents;

const int number_of_serial_circuits =
effective_configuration.serial_circuits_size();

MakegridCachedVectorPotential response_table_a;
response_table_a.parameters = makegrid_parameters;

response_table_a.a_r.resize(number_of_serial_circuits,
total_number_of_grid_points);
response_table_a.a_p.resize(number_of_serial_circuits,
total_number_of_grid_points);
response_table_a.a_z.resize(number_of_serial_circuits,
total_number_of_grid_points);

std::vector<absl::Status> status(number_of_serial_circuits);

for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
++circuit_index) {

MagneticConfiguration m_magnetic_configuration = effective_configuration;

Eigen::VectorXd currents_for_circuit;
currents_for_circuit.setZero(number_of_serial_circuits);
if (makegrid_parameters.normalize_by_currents) {
currents_for_circuit[circuit_index] = 1.0;
} else {
currents_for_circuit[circuit_index] = original_currents[circuit_index];
}

absl::Status set_currents_status =
SetCircuitCurrents(currents_for_circuit, m_magnetic_configuration);
if (!set_currents_status.ok()) {
status[circuit_index] = set_currents_status;
continue;
}

RowMatrix3Xd vector_potential =
RowMatrix3Xd::Zero(3, number_of_evaluation_points);

absl::Status vector_potential_status = VectorPotential(
m_magnetic_configuration, maybe_cylindrical_grid.value(),
vector_potential);
if (!vector_potential_status.ok()) {
status[circuit_index] = vector_potential_status;
continue;
}

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int linear_index = 0; linear_index < number_of_evaluation_points;
++linear_index) {
const double a_x = vector_potential(0, linear_index);
const double a_y = vector_potential(1, linear_index);
const double a_z = vector_potential(2, linear_index);

const size_t index_phi = linear_index / (num_z * num_r);
const double a_r = a_x * cos_phi[index_phi] + a_y * sin_phi[index_phi];
const double a_p = a_y * cos_phi[index_phi] - a_x * sin_phi[index_phi];

response_table_a.a_r(circuit_index, linear_index) = a_r;
response_table_a.a_p(circuit_index, linear_index) = a_p;
response_table_a.a_z(circuit_index, linear_index) = a_z;
}

if (makegrid_parameters.assume_stellarator_symmetry) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int linear_index = number_of_evaluation_points;
linear_index < total_number_of_grid_points; ++linear_index) {
const int idx_phi = linear_index / (num_z * num_r);
const int idx_z_r = linear_index % (num_z * num_r);
const int idx_z = idx_z_r / num_r;
const int idx_r = idx_z_r % num_r;

const int idx_phi_reversed = num_phi - idx_phi;
const int idx_z_reversed = num_z - 1 - idx_z;

const int linear_index_reversed =
(idx_phi_reversed * num_z + idx_z_reversed) * num_r + idx_r;

response_table_a.a_r(circuit_index, linear_index) =
-response_table_a.a_r(circuit_index, linear_index_reversed);
response_table_a.a_p(circuit_index, linear_index) =
response_table_a.a_p(circuit_index, linear_index_reversed);
response_table_a.a_z(circuit_index, linear_index) =
response_table_a.a_z(circuit_index, linear_index_reversed);
}
}

LOG(INFO) << absl::StrFormat("A %2d/%2d: done", circuit_index + 1,
number_of_serial_circuits);
}

for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
++circuit_index) {
if (!status[circuit_index].ok()) {
return status[circuit_index];
}
}

return response_table_a;
}

#define VMECPP_RETURN_IF_NETCDF_ERROR(nc_call)                               \
do {                                                                       \
const int nc_status = (nc_call);                                         \
if (nc_status != NC_NOERR) {                                             \
nc_close(ncid);                                                        \
return absl::InternalError(absl::StrFormat("could not write '%s': %s", \
makegrid_filename,          \
nc_strerror(nc_status)));   \
}                                                                        \
} while (0)

absl::Status WriteMakegridNetCDFFile(
const std::string& makegrid_filename,
const MakegridParameters& makegrid_parameters,
const Eigen::VectorXd& circuit_currents,
const MagneticFieldResponseTable& response_table_b,
const std::optional<MakegridCachedVectorPotential>& response_table_a) {
static constexpr int kStringSize = 30;
static constexpr char kRawCurrents = 'R';
static constexpr char kNormalizeByCurrents = 'S';

const int n_serial_circuits = static_cast<int>(response_table_b.b_r.rows());
if (n_serial_circuits <= 0) {
return absl::InvalidArgumentError(
"No magnetic field cache present to be written.");
}

const int n_circuit_currents = static_cast<int>(circuit_currents.size());
if (n_circuit_currents != n_serial_circuits) {
return absl::InvalidArgumentError(absl::StrFormat(
"number of provided circuit currents (%d) has to equal number of "
"serial circuits (%d)",
n_circuit_currents, n_serial_circuits));
}

int ncid = 0;
const int create_status =
nc_create(makegrid_filename.c_str(), NC_CLOBBER, &ncid);
if (create_status != NC_NOERR) {
return absl::InternalError(absl::StrFormat("could not create '%s': %s",
makegrid_filename,
nc_strerror(create_status)));
}

int id_dimension_stringsize = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_dim(ncid, "stringsize", kStringSize, &id_dimension_stringsize));

int id_dimension_external_coil_groups = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(nc_def_dim(ncid, "external_coil_groups",
n_serial_circuits,
&id_dimension_external_coil_groups));

int id_dimension_dim_00001 = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_dim(ncid, "dim_00001", 1, &id_dimension_dim_00001));

int id_dimension_external_coils = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(nc_def_dim(
ncid, "external_coils", n_serial_circuits, &id_dimension_external_coils));

int id_dimension_rad = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_dim(ncid, "rad", makegrid_parameters.number_of_r_grid_points,
&id_dimension_rad));

int id_dimension_zee = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_dim(ncid, "zee", makegrid_parameters.number_of_z_grid_points,
&id_dimension_zee));

int id_dimension_phi = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_dim(ncid, "phi", makegrid_parameters.number_of_phi_grid_points,
&id_dimension_phi));

int id_variable_ir = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_var(ncid, "ir", NC_INT, 0, nullptr, &id_variable_ir));

int id_variable_jz = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_var(ncid, "jz", NC_INT, 0, nullptr, &id_variable_jz));

int id_variable_kp = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_var(ncid, "kp", NC_INT, 0, nullptr, &id_variable_kp));

int id_variable_nfp = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_var(ncid, "nfp", NC_INT, 0, nullptr, &id_variable_nfp));

int id_variable_nextcur = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_var(ncid, "nextcur", NC_INT, 0, nullptr, &id_variable_nextcur));

int id_variable_rmin = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_var(ncid, "rmin", NC_DOUBLE, 0, nullptr, &id_variable_rmin));

int id_variable_rmax = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_var(ncid, "rmax", NC_DOUBLE, 0, nullptr, &id_variable_rmax));

int id_variable_zmin = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_var(ncid, "zmin", NC_DOUBLE, 0, nullptr, &id_variable_zmin));

int id_variable_zmax = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_def_var(ncid, "zmax", NC_DOUBLE, 0, nullptr, &id_variable_zmax));

int id_variable_coil_group = 0;
std::array<int, 2> coil_group_dimensions = {id_dimension_external_coil_groups,
id_dimension_stringsize};
VMECPP_RETURN_IF_NETCDF_ERROR(nc_def_var(ncid, "coil_group", NC_CHAR, 2,
coil_group_dimensions.data(),
&id_variable_coil_group));

int id_variable_mgrid_mode = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(nc_def_var(ncid, "mgrid_mode", NC_CHAR, 1,
&id_dimension_dim_00001,
&id_variable_mgrid_mode));

int id_variable_raw_coil_cur = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(nc_def_var(ncid, "raw_coil_cur", NC_DOUBLE, 1,
&id_dimension_external_coils,
&id_variable_raw_coil_cur));

std::vector<int> ids_variable_br(n_serial_circuits, 0);
std::vector<int> ids_variable_bp(n_serial_circuits, 0);
std::vector<int> ids_variable_bz(n_serial_circuits, 0);
std::vector<int> ids_variable_ar(n_serial_circuits, 0);
std::vector<int> ids_variable_ap(n_serial_circuits, 0);
std::vector<int> ids_variable_az(n_serial_circuits, 0);
for (int circuit_index = 0; circuit_index < n_serial_circuits;
++circuit_index) {
std::array<int, 3> grid_dimension = {id_dimension_phi, id_dimension_zee,
id_dimension_rad};

std::string br_name = absl::StrFormat("br_%03d", circuit_index + 1);
int id_variable_br = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(nc_def_var(ncid, br_name.c_str(), NC_DOUBLE,
3, grid_dimension.data(),
&id_variable_br));
ids_variable_br[circuit_index] = id_variable_br;

std::string bp_name = absl::StrFormat("bp_%03d", circuit_index + 1);
int id_variable_bp = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(nc_def_var(ncid, bp_name.c_str(), NC_DOUBLE,
3, grid_dimension.data(),
&id_variable_bp));
ids_variable_bp[circuit_index] = id_variable_bp;

std::string bz_name = absl::StrFormat("bz_%03d", circuit_index + 1);
int id_variable_bz = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(nc_def_var(ncid, bz_name.c_str(), NC_DOUBLE,
3, grid_dimension.data(),
&id_variable_bz));
ids_variable_bz[circuit_index] = id_variable_bz;

if (response_table_a.has_value()) {
std::string ar_name = absl::StrFormat("ar_%03d", circuit_index + 1);
int id_variable_ar = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(nc_def_var(ncid, ar_name.c_str(), NC_DOUBLE,
3, grid_dimension.data(),
&id_variable_ar));
ids_variable_ar[circuit_index] = id_variable_ar;

std::string ap_name = absl::StrFormat("ap_%03d", circuit_index + 1);
int id_variable_ap = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(nc_def_var(ncid, ap_name.c_str(), NC_DOUBLE,
3, grid_dimension.data(),
&id_variable_ap));
ids_variable_ap[circuit_index] = id_variable_ap;

std::string az_name = absl::StrFormat("az_%03d", circuit_index + 1);
int id_variable_az = 0;
VMECPP_RETURN_IF_NETCDF_ERROR(nc_def_var(ncid, az_name.c_str(), NC_DOUBLE,
3, grid_dimension.data(),
&id_variable_az));
ids_variable_az[circuit_index] = id_variable_az;
}
}

VMECPP_RETURN_IF_NETCDF_ERROR(nc_enddef(ncid));

VMECPP_RETURN_IF_NETCDF_ERROR(nc_put_var(
ncid, id_variable_ir, &(makegrid_parameters.number_of_r_grid_points)));

VMECPP_RETURN_IF_NETCDF_ERROR(nc_put_var(
ncid, id_variable_jz, &(makegrid_parameters.number_of_z_grid_points)));

VMECPP_RETURN_IF_NETCDF_ERROR(nc_put_var(
ncid, id_variable_kp, &(makegrid_parameters.number_of_phi_grid_points)));

VMECPP_RETURN_IF_NETCDF_ERROR(nc_put_var(
ncid, id_variable_nfp, &(makegrid_parameters.number_of_field_periods)));

VMECPP_RETURN_IF_NETCDF_ERROR(
nc_put_var(ncid, id_variable_nextcur, &n_serial_circuits));

VMECPP_RETURN_IF_NETCDF_ERROR(nc_put_var(
ncid, id_variable_rmin, &(makegrid_parameters.r_grid_minimum)));

VMECPP_RETURN_IF_NETCDF_ERROR(nc_put_var(
ncid, id_variable_rmax, &(makegrid_parameters.r_grid_maximum)));

VMECPP_RETURN_IF_NETCDF_ERROR(nc_put_var(
ncid, id_variable_zmin, &(makegrid_parameters.z_grid_minimum)));

VMECPP_RETURN_IF_NETCDF_ERROR(nc_put_var(
ncid, id_variable_zmax, &(makegrid_parameters.z_grid_maximum)));

std::string coil_group_names;

for (int circuit_index = 0; circuit_index < n_serial_circuits;
++circuit_index) {

std::string coil_group_name = absl::StrFormat("circuit_%d", circuit_index);

absl::StrAppend(&coil_group_names,
absl::StrFormat("%-30s", coil_group_name));
}
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_put_var(ncid, id_variable_coil_group, coil_group_names.c_str()));

if (makegrid_parameters.normalize_by_currents) {
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_put_var(ncid, id_variable_mgrid_mode, &kNormalizeByCurrents));
} else {
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_put_var(ncid, id_variable_mgrid_mode, &kRawCurrents));
}

VMECPP_RETURN_IF_NETCDF_ERROR(
nc_put_var(ncid, id_variable_raw_coil_cur, circuit_currents.data()));

for (int circuit_index = 0; circuit_index < n_serial_circuits;
++circuit_index) {
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_put_var(ncid, ids_variable_br[circuit_index],
response_table_b.b_r.row(circuit_index).data()));
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_put_var(ncid, ids_variable_bp[circuit_index],
response_table_b.b_p.row(circuit_index).data()));
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_put_var(ncid, ids_variable_bz[circuit_index],
response_table_b.b_z.row(circuit_index).data()));

if (response_table_a.has_value()) {
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_put_var(ncid, ids_variable_ar[circuit_index],
response_table_a->a_r.row(circuit_index).data()));
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_put_var(ncid, ids_variable_ap[circuit_index],
response_table_a->a_p.row(circuit_index).data()));
VMECPP_RETURN_IF_NETCDF_ERROR(
nc_put_var(ncid, ids_variable_az[circuit_index],
response_table_a->a_z.row(circuit_index).data()));
}
}

const int close_status = nc_close(ncid);
if (close_status != NC_NOERR) {
return absl::InternalError(absl::StrFormat("could not close '%s': %s",
makegrid_filename,
nc_strerror(close_status)));
}

return absl::OkStatus();
}

#undef VMECPP_RETURN_IF_NETCDF_ERROR

}

// source: vmecpp/common/sizes/sizes.cc

#include <algorithm>
#include <iostream>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"

namespace vmecpp {

Sizes::Sizes(const VmecINDATA& id)
: Sizes(id.lasym, id.nfp, id.mpol, id.ntor, id.ntheta, id.nzeta,
id.mpol_geometry, id.ntor_geometry) {}

Sizes::Sizes(bool lasym, int nfp, int mpol, int ntor, int ntheta, int nzeta,
int mpol_geometry, int ntor_geometry)
: lasym(lasym),
nfp(nfp),
mpol(mpol),
ntor(ntor),
mpolGeometry((mpol_geometry < 1 || mpol_geometry > mpol) ? mpol
: mpol_geometry),
ntorGeometry((ntor_geometry < 0 || ntor_geometry > ntor) ? ntor
: ntor_geometry),
ntheta(ntheta),
nZeta(nzeta) {
computeDerivedSizes();
}

void Sizes::computeDerivedSizes() {

CHECK_GE(nfp, 1) << "input variable 'nfp' needs to be >= 1, but is " << nfp;

CHECK_GE(mpol, 1) << "input variable 'mpol' needs to be >= 1, but is "
<< mpol;

CHECK_GE(ntor, 0) << "input variable 'ntor' needs to be >= 0, but is "
<< ntor;

if (ntheta < 2 * mpol + 6) {
ntheta = 2 * mpol + 6;
#ifdef DEBUG

std::cout << absl::StrFormat(
"adjusting 'ntheta' to %d in order to satisfy Nyquist criterion\n",
ntheta);
#endif
}

if (ntor == 0 && nZeta < 1) {

nZeta = 1;
}

if (ntor > 0) {

if (nZeta < 2 * ntor + 4) {
nZeta = 2 * ntor + 4;
#ifdef DEBUG

std::cout << absl::StrFormat(
"adjusting 'nzeta' to %d in order to satisfy Nyquist criterion\n",
nZeta);
#endif
}
}

lthreed = (ntor > 0);

num_basis = 1;
if (lthreed) {
num_basis *= 2;
}
if (lasym) {
num_basis *= 2;
}

nThetaEven = 2 * (ntheta / 2);

nThetaReduced = nThetaEven / 2 + 1;

if (lasym) {
nThetaEff = nThetaEven;
} else {

nThetaEff = nThetaReduced;
}

nZnT = nZeta * nThetaEff;

double dnorm3 = 1.0 / (nZeta * (nThetaReduced - 1));
if (lasym) {
dnorm3 = 1.0 / (nZeta * nThetaEven);
}

wInt.resize(nThetaEff);
for (int l = 0; l < nThetaEff; ++l) {
wInt[l] = dnorm3;
if (!lasym && (l == 0 || l == nThetaReduced - 1)) {

wInt[l] /= 2.0;
}
}

mnsize = mpol * (ntor + 1);

mnmax = (ntor + 1) + (mpol - 1) * (2 * ntor + 1);

int mnyq0 = nThetaEven / 2;
int nnyq0 = nZeta / 2;

mnyq2 = std::max(0, std::max(2 * mnyq0, 2 * (mpol - 1)));
nnyq2 = std::max(0, std::max(2 * nnyq0, 2 * ntor));

mnmax_nyq = nnyq2 / 2 + 1 + mnyq2 / 2 * (nnyq2 + 1);

mnyq = mnyq2 / 2;
nnyq = nnyq2 / 2;
}

}

// source: vmecpp/common/util/util.cc

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace vmecpp {

int VmecStatusCode(const VmecStatus vmec_status) {

return static_cast<std::underlying_type<VmecStatus>::type>(vmec_status);
}

std::string VmecStatusAsString(const VmecStatus vmec_status) {
switch (vmec_status) {
case VmecStatus::NORMAL_TERMINATION:
return "NORMAL_TERMINATION";
case VmecStatus::BAD_JACOBIAN:
return "BAD_JACOBIAN: the Jacobian of the flux-surface geometry "
"changed sign, i.e. flux surfaces overlap. This can happen "
"with a poor initial guess for the magnetic axis or an "
"initial boundary shape that is too far from an equilibrium";
case VmecStatus::JACOBIAN_75_TIMES_BAD:
return "JACOBIAN_75_TIMES_BAD The jacobian factor of the geometry "
"repeatedly "
"became negative, indicating that flux surfaces become "
"self-intersecting. This "
"can mean that your prescribed boundary is too shaped for VMEC to "
"resolve ";
case VmecStatus::UNRECOVERABLE_ERROR:
return "UNRECOVERABLE_ERROR: a physical inconsistency was detected "
"in the MHD model (e.g. a degenerate flux-surface geometry or "
"a free-boundary current mismatch) that the solver could not "
"recover from";
case VmecStatus::SUCCESSFUL_TERMINATION:
return "SUCCESSFUL_TERMINATION";
}

return "UNKNOWN??" + std::to_string(static_cast<int>(vmec_status));
}

int signum(int x) {

return (x > 0) - (x < 0);
}

void TridiagonalSolveSerial(std::span<double> m_a, std::span<double> m_d,
std::span<double> m_b, double* m_c_data,
int c_stride, int jMin, int jMax, int nRHS) {

auto c = [m_c_data, c_stride](int k, int j) -> double& {
return m_c_data[k * c_stride + j];
};

for (int j = 0; j < jMin; ++j) {
m_a[j] = 0.0;
m_d[j] = 1.0;
m_b[j] = 0.0;
for (int k = 0; k < nRHS; ++k) {
c(k, j) = 0.0;
}
}

m_a[jMin] /= m_d[jMin];

for (int j = jMin + 1; j < jMax - 1; ++j) {
const double denominator = m_d[j] - m_a[j - 1] * m_b[j];

m_a[j] /= denominator;
}

for (int k = 0; k < nRHS; ++k) {

c(k, jMin) /= m_d[jMin];

for (int j = jMin + 1; j < jMax; ++j) {
const double denominator = m_d[j] - m_a[j - 1] * m_b[j];

c(k, j) = (c(k, j) - c(k, j - 1) * m_b[j]) / denominator;
}

for (int j = jMax - 2; j > jMin - 1; --j) {
c(k, j) -= m_a[j] * c(k, j + 1);
}
}
}

void TridiagonalSolveOpenMP(
std::vector<double>& m_ar, std::vector<double>& m_dr,
std::vector<double>& m_br, std::vector<std::span<double>>& m_cr,
std::vector<double>& m_az, std::vector<double>& m_dz,
std::vector<double>& m_bz, std::vector<std::span<double>>& m_cz,
const std::vector<int>& jMin, int jMax, int mnmax, int nRHS,
std::vector<std::mutex>& m_mutices, int ncpu, int myid, int nsMinF,
int nsMaxF, std::vector<double>& m_handover_ar,
std::vector<std::vector<double>>& m_handover_cr,
std::vector<double>& m_handover_az,
std::vector<std::vector<double>>& m_handover_cz) {
#ifdef _OPENMP
#pragma omp barrier
#endif

if (myid == 0) {

for (int j = 0; j < nsMaxF; ++j) {
for (int mn = 0; mn < mnmax; ++mn) {
if (j < jMin[mn]) {
int idx_mn = (j - nsMinF) * mnmax + mn;
m_ar[idx_mn] = 0.0;
m_az[idx_mn] = 0.0;
m_dr[idx_mn] = 1.0;
m_dz[idx_mn] = 1.0;
m_br[idx_mn] = 0.0;
m_bz[idx_mn] = 0.0;
}
}
}
for (int k = 0; k < nRHS; ++k) {
for (int j = 0; j < nsMaxF; ++j) {
for (int mn = 0; mn < mnmax; ++mn) {
if (j < jMin[mn]) {
int idx_mn = (j - nsMinF) * mnmax + mn;
m_cr[k][idx_mn] = 0.0;
m_cz[k][idx_mn] = 0.0;
}
}
}
}

for (int mn = 0; mn < mnmax; ++mn) {
int idx_mn = (jMin[mn] - nsMinF) * mnmax + mn;

m_ar[idx_mn] /= m_dr[idx_mn];
m_az[idx_mn] /= m_dz[idx_mn];
for (int k = 0; k < nRHS; ++k) {
m_cr[k][idx_mn] /= m_dr[idx_mn];
m_cz[k][idx_mn] /= m_dz[idx_mn];
}
}
}

if (myid + 1 < ncpu) {
m_mutices[myid + 1].lock();
}

#ifdef _OPENMP
#pragma omp barrier
#endif

m_mutices[myid].lock();

for (int j = nsMinF; j < std::min(jMax, nsMaxF); ++j) {
for (int mn = 0; mn < mnmax; ++mn) {

if (j < jMin[mn] + 1) {
continue;
}

int idx_mn_0 = (j - nsMinF) * mnmax + mn;
int idx_mn_m = (j - 1 - nsMinF) * mnmax + mn;

double prev_m_ar;
double prev_az;
std::vector<std::span<double>> prev_cr(m_handover_cr.size());
std::vector<std::span<double>> prev_cz(m_handover_cz.size());
int prev_c_idx;
if (j == nsMinF) {

prev_m_ar = m_handover_ar[mn];
prev_az = m_handover_az[mn];
for (std::size_t i = 0; i < m_handover_cr.size(); ++i) {
prev_cr[i] = m_handover_cr[i];
prev_cz[i] = m_handover_cz[i];
}

prev_c_idx = mn;
} else {

prev_m_ar = m_ar[idx_mn_m];
prev_az = m_az[idx_mn_m];
prev_c_idx = idx_mn_m;
for (std::size_t i = 0; i < m_cr.size(); ++i) {
prev_cr[i] = m_cr[i];
prev_cz[i] = m_cz[i];
}
}

double denom_r = m_dr[idx_mn_0] - prev_m_ar * m_br[idx_mn_0];
double denom_z = m_dz[idx_mn_0] - prev_az * m_bz[idx_mn_0];

if (j < jMax - 1) {
m_ar[idx_mn_0] /= denom_r;
m_az[idx_mn_0] /= denom_z;
}

for (int k = 0; k < nRHS; ++k) {
m_cr[k][idx_mn_0] =
(m_cr[k][idx_mn_0] - prev_cr[k][prev_c_idx] * m_br[idx_mn_0]) /
denom_r;
m_cz[k][idx_mn_0] =
(m_cz[k][idx_mn_0] - prev_cz[k][prev_c_idx] * m_bz[idx_mn_0]) /
denom_z;
}

if (j == nsMaxF - 1) {

m_handover_ar[mn] = m_ar[idx_mn_0];
m_handover_az[mn] = m_az[idx_mn_0];
for (int k = 0; k < nRHS; ++k) {
m_handover_cr[k][mn] = m_cr[k][idx_mn_0];
m_handover_cz[k][mn] = m_cz[k][idx_mn_0];
}
}
}
}

m_mutices[myid].unlock();

if (myid + 1 < ncpu) {

m_mutices[myid + 1].unlock();
}

if (myid > 0) {
m_mutices[myid - 1].lock();
}

#ifdef _OPENMP
#pragma omp barrier
#endif

m_mutices[myid].lock();

for (int j = std::min(jMax - 2, nsMaxF - 1); j >= nsMinF; --j) {
for (int mn = 0; mn < mnmax; ++mn) {

if (j < jMin[mn]) {
continue;
}

int idx_mn_p = (j + 1 - nsMinF) * mnmax + mn;
int idx_mn_0 = (j - nsMinF) * mnmax + mn;

std::vector<std::span<double>> prev_cr(m_handover_cr.size());
std::vector<std::span<double>> prev_cz(m_handover_cz.size());
int prev_c_idx;
if (j == nsMaxF - 1) {

for (std::size_t i = 0; i < m_handover_cr.size(); ++i) {
prev_cr[i] = m_handover_cr[i];
prev_cz[i] = m_handover_cz[i];
}

prev_c_idx = mn;
} else {

for (std::size_t i = 0; i < m_cr.size(); ++i) {
prev_cr[i] = m_cr[i];
prev_cz[i] = m_cz[i];
}
prev_c_idx = idx_mn_p;
}

for (int k = 0; k < nRHS; ++k) {
m_cr[k][idx_mn_0] -= m_ar[idx_mn_0] * prev_cr[k][prev_c_idx];
m_cz[k][idx_mn_0] -= m_az[idx_mn_0] * prev_cz[k][prev_c_idx];
}

if (j == nsMinF) {

for (int k = 0; k < nRHS; ++k) {
m_handover_cr[k][mn] = m_cr[k][idx_mn_0];
m_handover_cz[k][mn] = m_cz[k][idx_mn_0];
}
}
}
}

m_mutices[myid].unlock();

if (myid > 0) {

m_mutices[myid - 1].unlock();
}

#ifdef _OPENMP
#pragma omp barrier
#endif
}

int vmec_adjust_num_threads(const int max_threads,
const int num_surfaces_to_distribute) {

int num_threads = std::min(max_threads, num_surfaces_to_distribute / 2);

#ifdef _OPENMP

omp_set_dynamic(0);
#endif

return num_threads;
}

int vmec_adjust_vacuum_num_threads(const int max_threads, const int n_znt) {

return std::min(max_threads, n_znt);
}

}

// source: vmecpp/common/vmec_indata/boundary_from_json.cc

// header: vmecpp/common/vmec_indata/boundary_from_json.h

#ifndef VMECPP_COMMON_VMEC_INDATA_BOUNDARY_FROM_JSON_H_
#define VMECPP_COMMON_VMEC_INDATA_BOUNDARY_FROM_JSON_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace vmecpp {

struct BoundaryCoefficient {

int m;

int n;

double value;

static absl::StatusOr<std::optional<std::vector<BoundaryCoefficient> > >
FromJson(const nlohmann::json& j, const std::string& name);
};

std::vector<double> BoundaryFromJson(const nlohmann::json& json,
const std::string& key, int mpol,
int ntor);

}

#endif

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "nlohmann/json.hpp"

namespace vmecpp {

using nlohmann::json;

using json_io::JsonReadBool;
using json_io::JsonReadDouble;
using json_io::JsonReadInt;
using json_io::JsonReadString;
using json_io::JsonReadVectorDouble;
using json_io::JsonReadVectorInt;

absl::StatusOr<std::optional<std::vector<BoundaryCoefficient> > >
BoundaryCoefficient::FromJson(const json& j, const std::string& name) {
if (!j.contains(name)) {

return std::nullopt;
}

if (!j[name].is_array()) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON element '%s' is not an array", name));
}

std::vector<BoundaryCoefficient> entries;
int i = 0;
for (const json& entry : j[name]) {
if (!entry.is_object()) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON entry '%s'[%d] is not an object", name, i));
}

auto m = JsonReadInt(entry, "m");
if (!m.ok()) {
return m.status();
}
if (!m->has_value()) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON entry '%s'[%d] has no 'm'", name, i));
}
if (m->value() < 0) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON entry '%s'[%d] has m = %d, but m cannot be "
"negative",
name, i, m->value()));
}

auto n = JsonReadInt(entry, "n");
if (!n.ok()) {
return n.status();
}
if (!n->has_value()) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON entry '%s'[%d] has no 'n'", name, i));
}

auto value = JsonReadDouble(entry, "value");
if (!value.ok()) {
return value.status();
}
if (!value->has_value()) {
return absl::InvalidArgumentError(
absl::StrFormat("JSON entry '%s'[%d] has no 'value'", name, i));
}

BoundaryCoefficient boundary_coefficient = {
m->value(),  n->value(),  value->value()};
entries.push_back(boundary_coefficient);

i++;
}

return entries;
}

std::vector<double> BoundaryFromJson(const nlohmann::json& json,
const std::string& key, int mpol,
int ntor) {
std::vector<double> coeffs(mpol * (2 * ntor + 1));

const auto maybe_entries = vmecpp::BoundaryCoefficient::FromJson(json, key);
CHECK_OK(maybe_entries);
CHECK(maybe_entries->has_value());

const std::vector<vmecpp::BoundaryCoefficient> entries =
maybe_entries->value();
for (const vmecpp::BoundaryCoefficient& entry : entries) {

if (entry.m < 0 || entry.m >= mpol || entry.n < -ntor || entry.n > ntor) {

continue;
}

const int index_along_n = ntor + entry.n;
const int flat_index = entry.m * (2 * ntor + 1) + index_along_n;
coeffs[flat_index] = entry.value;
}

return coeffs;
}

}

// source: vmecpp/common/vmec_indata/vmec_indata.cc

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>

#include "H5Cpp.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "nlohmann/json.hpp"

// header: vmecpp/vmec/profile_parameterization_data/profile_parameterization_data.h

#ifndef VMECPP_VMEC_PROFILE_PARAMETERIZATION_DATA_PROFILE_PARAMETERIZATION_DATA_H_
#define VMECPP_VMEC_PROFILE_PARAMETERIZATION_DATA_PROFILE_PARAMETERIZATION_DATA_H_

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace vmecpp {

#define NUM_PARAM 23

enum class ProfileType : std::uint8_t { PRESSURE, CURRENT, IOTA };

struct AllowedFor {
bool pres;
bool curr;
bool iota;
};

class ProfileParameterizationData {
public:
ProfileParameterizationData(const std::string& name, bool allowedForPres,
bool allowedForCurr, bool allowedForIota,
bool needsSplineData);

const std::string& Name() const;
bool NeedsSplineData() const;
AllowedFor IsAllowedFor() const;

private:
const std::string name_;
bool needsSplineData_;
AllowedFor allowedFor_;
};

enum class ProfileParameterization : std::uint8_t {
INVALID_PARAM = 0,
POWER_SERIES = 1,
POWER_SERIES_I = 2,
GAUSS_TRUNC = 3,
SUM_ATAN = 4,
TWO_LORENTZ = 5,
TWO_POWER = 6,
TWO_POWER_GS = 7,
AKIMA_SPLINE = 8,
AKIMA_SPLINE_I = 9,
AKIMA_SPLINE_IP = 10,
CUBIC_SPLINE = 11,
CUBIC_SPLINE_I = 12,
CUBIC_SPLINE_IP = 13,
PEDESTAL = 14,
RATIONAL = 15,
LINE_SEGMENT = 16,
LINE_SEGMENT_I = 17,
LINE_SEGMENT_IP = 18,
NICE_QUADRATIC = 19,
SUM_COSSQ_S = 20,
SUM_COSSQ_SQRTS = 21,
SUM_COSSQ_S_FREE = 22
};

const std::vector<ProfileParameterizationData>& AllProfileParameterizations();

const ProfileParameterizationData* FindProfileParameterization(
const std::string& name);

bool IsProfileParameterizationAllowedFor(const std::string& name,
ProfileType type);

}

#endif

namespace {
[[noreturn]] void ErrorToException(const absl::Status& status,
const std::string& context) {
const std::string msg =
"There was an error " + context + ":\n" + std::string(status.message());
throw std::runtime_error(msg);
}

std::string ProfileTypeName(vmecpp::ProfileType profile_type) {
switch (profile_type) {
case vmecpp::ProfileType::PRESSURE:
return "mass/pressure";
case vmecpp::ProfileType::CURRENT:
return "current";
case vmecpp::ProfileType::IOTA:
return "iota";
}
return "unknown";
}

absl::Status CheckProfile(const std::string& type_key,
const std::string& type_name,
vmecpp::ProfileType profile_type,
const std::string& aux_key,
const Eigen::VectorXd& aux_s,
const Eigen::VectorXd& aux_f) {
const vmecpp::ProfileParameterizationData* const parameterization =
vmecpp::FindProfileParameterization(type_name);
if (parameterization == nullptr) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable '%s' is '%s', which is not a known profile "
"parameterization\n",
type_key, type_name));
}

if (!vmecpp::IsProfileParameterizationAllowedFor(type_name, profile_type)) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable '%s' is '%s', which cannot be used for the %s "
"profile\n",
type_key, type_name, ProfileTypeName(profile_type)));
}

if (parameterization->NeedsSplineData()) {
if (aux_s.size() == 0 || aux_f.size() == 0) {
return absl::InvalidArgumentError(absl::StrFormat(
"'%s' is '%s', which is a spline profile, so '%s_aux_s' and "
"'%s_aux_f' must be given\n",
type_key, type_name, aux_key, aux_key));
}
if (aux_s.size() != aux_f.size()) {
return absl::InvalidArgumentError(absl::StrFormat(
"'%s_aux_s' and '%s_aux_f' must have the same number of entries, "
"but have %d and %d\n",
aux_key, aux_key, aux_s.size(), aux_f.size()));
}
}

return absl::OkStatus();
}
}

namespace vmecpp {

using nlohmann::json;

using json_io::JsonParse;
using json_io::JsonReadBool;
using json_io::JsonReadDouble;
using json_io::JsonReadInt;
using json_io::JsonReadString;
using json_io::JsonReadVectorDouble;
using json_io::JsonReadVectorInt;

int FreeBoundaryMethodCode(FreeBoundaryMethod free_boundary_method) {

return static_cast<std::underlying_type_t<FreeBoundaryMethod>>(
free_boundary_method);
}

absl::StatusOr<FreeBoundaryMethod> FreeBoundaryMethodFromString(
const std::string& free_boundary_method_string) {
if (free_boundary_method_string == "nestor") {
return FreeBoundaryMethod::NESTOR;
} else if (free_boundary_method_string == "only_coils") {
return FreeBoundaryMethod::ONLY_COILS;
} else if (free_boundary_method_string == "biest") {
return FreeBoundaryMethod::BIEST;
}
return absl::NotFoundError(absl::StrCat("free boundary method named '",
free_boundary_method_string,
"' not known"));
}

std::string ToString(FreeBoundaryMethod free_boundary_method) {
switch (free_boundary_method) {
case FreeBoundaryMethod::NESTOR:
return "nestor";
case FreeBoundaryMethod::ONLY_COILS:
return "only_coils";
case FreeBoundaryMethod::BIEST:
return "biest";
default:
LOG(FATAL)
<< "no string conversion implemented yet for FreeBoundaryMethod code "
<< FreeBoundaryMethodCode(free_boundary_method);
}
}

int IterationStyleCode(IterationStyle iteration_style) {

return static_cast<std::underlying_type_t<IterationStyle>>(iteration_style);
}

absl::StatusOr<IterationStyle> IterationStyleFromString(
const std::string& iteration_style_string) {
if (iteration_style_string == "vmec_8_52") {
return IterationStyle::VMEC_8_52;
} else if (iteration_style_string == "parvmec") {
return IterationStyle::PARVMEC;
}
return absl::NotFoundError(absl::StrCat(
"iteration style named '", iteration_style_string, "' not known"));
}

std::string ToString(IterationStyle iteration_style) {
switch (iteration_style) {
case IterationStyle::VMEC_8_52:
return "vmec_8_52";
case IterationStyle::PARVMEC:
return "parvmec";
default:
LOG(FATAL)
<< "no string conversion implemented yet for IterationStyle code "
<< IterationStyleCode(iteration_style);
}
}

VmecINDATA::VmecINDATA() {

lasym = false;
nfp = 1;
mpol = 6;
ntor = 0;
mpol_geometry = -1;
ntor_geometry = -1;
ntheta = 0;
nzeta = 0;

ns_array.resize(1);
ns_array[0] = kNsDefault;
ftol_array.resize(1);
ftol_array[0] = kFTolDefault;
niter_array.resize(1);
niter_array[0] = kNIterDefault;

phiedge = 1.0;
ncurr = 0;

pmass_type = "power_series";

pres_scale = 1.0;
gamma = 0.0;
spres_ped = 1.0;

piota_type = "power_series";

pcurr_type = "power_series";

curtor = 0.0;
bloat = 1.0;

lfreeb = false;
mgrid_file = "NONE";

nvacskip = 1;
signgs = -1;
free_boundary_method = FreeBoundaryMethod::NESTOR;

nstep = 10;
aphi.resize(1);
aphi[0] = 1.0;
delt = 1.0;
tcon0 = 1.0;
lforbal = false;
iteration_style = IterationStyle::VMEC_8_52;
return_outputs_even_if_not_converged = false;

raxis_c.setZero(ntor + 1);
zaxis_s.setZero(ntor + 1);
if (lasym) {
raxis_s.emplace().setZero(ntor + 1);
zaxis_c.emplace().setZero(ntor + 1);
}

rbc.setZero(mpol, 2 * ntor + 1);
zbs.setZero(mpol, 2 * ntor + 1);
if (lasym) {
rbs.emplace().setZero(mpol, 2 * ntor + 1);
zbc.emplace().setZero(mpol, 2 * ntor + 1);
}
}

void VmecINDATA::SetMpolNtor(int new_mpol, int new_ntor) {
using Eigen::VectorXd;

if (lasym) {
if (!raxis_s.has_value()) raxis_s.emplace().setZero(ntor + 1);
if (!zaxis_c.has_value()) zaxis_c.emplace().setZero(ntor + 1);
if (!rbs.has_value()) rbs.emplace().setZero(mpol, 2 * ntor + 1);
if (!zbc.has_value()) zbc.emplace().setZero(mpol, 2 * ntor + 1);
}

const bool both_same_as_before = (new_mpol == mpol && new_ntor == ntor);
if (both_same_as_before) {
return;
}

VectorXd old_axis_fc = raxis_c;
const auto shortest_range = Eigen::seq(0, std::min(ntor, new_ntor));

raxis_c = VectorXd::Zero(new_ntor + 1);

raxis_c(shortest_range) = old_axis_fc(shortest_range);

old_axis_fc = zaxis_s;
zaxis_s = VectorXd::Zero(new_ntor + 1);
zaxis_s(shortest_range) = old_axis_fc(shortest_range);

if (lasym) {
old_axis_fc = raxis_s.value();
raxis_s = VectorXd::Zero(new_ntor + 1);
(*raxis_s)(shortest_range) = old_axis_fc(shortest_range);

old_axis_fc = zaxis_c.value();
zaxis_c = VectorXd::Zero(new_ntor + 1);
(*zaxis_c)(shortest_range) = old_axis_fc(shortest_range);
}

auto resized_2d_coeff = [this, new_mpol, new_ntor](const auto& coeff) {
const int new_nmax = (2 * new_ntor) + 1;
RowMatrixXd resized_coeff = RowMatrixXd::Zero(new_mpol, new_nmax);

const int smaller_ntor = std::min(ntor, new_ntor);
const int smaller_mpol = std::min(mpol, new_mpol);
for (int m = 0; m < smaller_mpol; ++m) {
for (int n = -smaller_ntor; n <= smaller_ntor; ++n) {
resized_coeff(m, n + new_ntor) = coeff(m, n + ntor);
}
}

return resized_coeff;
};

rbc = resized_2d_coeff(rbc);
zbs = resized_2d_coeff(zbs);

if (lasym) {
rbs = resized_2d_coeff(rbs.value());
zbc = resized_2d_coeff(zbc.value());
}

mpol = new_mpol;
ntor = new_ntor;
}

absl::Status VmecINDATA::WriteTo(H5::H5File& file) const {
using hdf5_io::WriteH5Dataset;
file.createGroup("/indata");

WriteH5Dataset(lasym, "/indata/lasym", file);
WriteH5Dataset(nfp, "/indata/nfp", file);
WriteH5Dataset(mpol, "/indata/mpol", file);
WriteH5Dataset(ntor, "/indata/ntor", file);
WriteH5Dataset(mpol_geometry, "/indata/mpol_geometry", file);
WriteH5Dataset(ntor_geometry, "/indata/ntor_geometry", file);
WriteH5Dataset(ntheta, "/indata/ntheta", file);
WriteH5Dataset(nzeta, "/indata/nzeta", file);
WriteH5Dataset(phiedge, "/indata/phiedge", file);
WriteH5Dataset(ncurr, "/indata/ncurr", file);
WriteH5Dataset(pmass_type, "/indata/pmass_type", file);
WriteH5Dataset(pres_scale, "/indata/pres_scale", file);
WriteH5Dataset(gamma, "/indata/gamma", file);
WriteH5Dataset(spres_ped, "/indata/spres_ped", file);
WriteH5Dataset(piota_type, "/indata/piota_type", file);
WriteH5Dataset(pcurr_type, "/indata/pcurr_type", file);
WriteH5Dataset(curtor, "/indata/curtor", file);
WriteH5Dataset(bloat, "/indata/bloat", file);
WriteH5Dataset(lfreeb, "/indata/lfreeb", file);
WriteH5Dataset(mgrid_file, "/indata/mgrid_file", file);
WriteH5Dataset(nvacskip, "/indata/nvacskip", file);
WriteH5Dataset(signgs, "/indata/signgs", file);

WriteH5Dataset(ToString(free_boundary_method), "/indata/free_boundary_method",
file);
WriteH5Dataset(ToString(iteration_style), "/indata/iteration_style", file);

WriteH5Dataset(nstep, "/indata/nstep", file);
WriteH5Dataset(delt, "/indata/delt", file);
WriteH5Dataset(tcon0, "/indata/tcon0", file);
WriteH5Dataset(lforbal, "/indata/lforbal", file);
WriteH5Dataset(return_outputs_even_if_not_converged,
"/indata/return_outputs_even_if_not_converged", file);

WriteH5Dataset(ns_array, "/indata/ns_array", file);
WriteH5Dataset(ftol_array, "/indata/ftol_array", file);
WriteH5Dataset(niter_array, "/indata/niter_array", file);
WriteH5Dataset(am, "/indata/am", file);
WriteH5Dataset(am_aux_s, "/indata/am_aux_s", file);
WriteH5Dataset(am_aux_f, "/indata/am_aux_f", file);
WriteH5Dataset(ai, "/indata/ai", file);
WriteH5Dataset(ai_aux_s, "/indata/ai_aux_s", file);
WriteH5Dataset(ai_aux_f, "/indata/ai_aux_f", file);
WriteH5Dataset(ac, "/indata/ac", file);
WriteH5Dataset(ac_aux_s, "/indata/ac_aux_s", file);
WriteH5Dataset(ac_aux_f, "/indata/ac_aux_f", file);
WriteH5Dataset(extcur, "/indata/extcur", file);
WriteH5Dataset(aphi, "/indata/aphi", file);
WriteH5Dataset(raxis_c, "/indata/raxis_c", file);
WriteH5Dataset(zaxis_s, "/indata/zaxis_s", file);
if (lasym) {
WriteH5Dataset(*raxis_s, "/indata/raxis_s", file);
WriteH5Dataset(*zaxis_c, "/indata/zaxis_c", file);
}

WriteH5Dataset(rbc, "/indata/rbc", file);
WriteH5Dataset(zbs, "/indata/zbs", file);
if (lasym) {
WriteH5Dataset(*rbs, "/indata/rbs", file);
WriteH5Dataset(*zbc, "/indata/zbc", file);
}

return absl::OkStatus();
}

absl::Status VmecINDATA::LoadInto(VmecINDATA& m_indata, H5::H5File& from_file) {
using hdf5_io::ReadH5Dataset;

ReadH5Dataset(m_indata.lasym, "/indata/lasym", from_file);
ReadH5Dataset(m_indata.nfp, "/indata/nfp", from_file);
ReadH5Dataset(m_indata.mpol, "/indata/mpol", from_file);
ReadH5Dataset(m_indata.ntor, "/indata/ntor", from_file);

if (from_file.nameExists("/indata/mpol_geometry")) {
ReadH5Dataset(m_indata.mpol_geometry, "/indata/mpol_geometry", from_file);
}
if (from_file.nameExists("/indata/ntor_geometry")) {
ReadH5Dataset(m_indata.ntor_geometry, "/indata/ntor_geometry", from_file);
}
ReadH5Dataset(m_indata.ntheta, "/indata/ntheta", from_file);
ReadH5Dataset(m_indata.nzeta, "/indata/nzeta", from_file);
ReadH5Dataset(m_indata.phiedge, "/indata/phiedge", from_file);
ReadH5Dataset(m_indata.ncurr, "/indata/ncurr", from_file);
ReadH5Dataset(m_indata.pmass_type, "/indata/pmass_type", from_file);
ReadH5Dataset(m_indata.pres_scale, "/indata/pres_scale", from_file);
ReadH5Dataset(m_indata.gamma, "/indata/gamma", from_file);
ReadH5Dataset(m_indata.spres_ped, "/indata/spres_ped", from_file);
ReadH5Dataset(m_indata.piota_type, "/indata/piota_type", from_file);
ReadH5Dataset(m_indata.pcurr_type, "/indata/pcurr_type", from_file);
ReadH5Dataset(m_indata.curtor, "/indata/curtor", from_file);
ReadH5Dataset(m_indata.bloat, "/indata/bloat", from_file);
ReadH5Dataset(m_indata.lfreeb, "/indata/lfreeb", from_file);
ReadH5Dataset(m_indata.mgrid_file, "/indata/mgrid_file", from_file);
ReadH5Dataset(m_indata.nvacskip, "/indata/nvacskip", from_file);
if (from_file.nameExists("/indata/signgs")) {
ReadH5Dataset(m_indata.signgs, "/indata/signgs", from_file);
}

std::string fbdy_method_str;
ReadH5Dataset(fbdy_method_str, "/indata/free_boundary_method", from_file);
const auto maybe_fbdy_method = FreeBoundaryMethodFromString(fbdy_method_str);
if (!maybe_fbdy_method.ok()) {
return maybe_fbdy_method.status();
}
m_indata.free_boundary_method = maybe_fbdy_method.value();

if (H5Lexists(from_file.getId(), "/indata/iteration_style", 0) == 1) {
std::string iteration_style_str;
ReadH5Dataset(iteration_style_str, "/indata/iteration_style", from_file);
const auto maybe_iteration_style =
IterationStyleFromString(iteration_style_str);
if (!maybe_iteration_style.ok()) {
return maybe_iteration_style.status();
}
m_indata.iteration_style = maybe_iteration_style.value();
} else {

m_indata.iteration_style = IterationStyle::VMEC_8_52;
}

ReadH5Dataset(m_indata.nstep, "/indata/nstep", from_file);
ReadH5Dataset(m_indata.delt, "/indata/delt", from_file);
ReadH5Dataset(m_indata.tcon0, "/indata/tcon0", from_file);
ReadH5Dataset(m_indata.lforbal, "/indata/lforbal", from_file);

if (H5Lexists(from_file.getId(),
"/indata/return_outputs_even_if_not_converged", 0) == 1) {
ReadH5Dataset(m_indata.return_outputs_even_if_not_converged,
"/indata/return_outputs_even_if_not_converged", from_file);
} else {
m_indata.return_outputs_even_if_not_converged = false;
}

ReadH5Dataset(m_indata.ns_array, "/indata/ns_array", from_file);
ReadH5Dataset(m_indata.ftol_array, "/indata/ftol_array", from_file);
ReadH5Dataset(m_indata.niter_array, "/indata/niter_array", from_file);
ReadH5Dataset(m_indata.am, "/indata/am", from_file);
ReadH5Dataset(m_indata.am_aux_s, "/indata/am_aux_s", from_file);
ReadH5Dataset(m_indata.am_aux_f, "/indata/am_aux_f", from_file);
ReadH5Dataset(m_indata.ai, "/indata/ai", from_file);
ReadH5Dataset(m_indata.ai_aux_s, "/indata/ai_aux_s", from_file);
ReadH5Dataset(m_indata.ai_aux_f, "/indata/ai_aux_f", from_file);
ReadH5Dataset(m_indata.ac, "/indata/ac", from_file);
ReadH5Dataset(m_indata.ac_aux_s, "/indata/ac_aux_s", from_file);
ReadH5Dataset(m_indata.ac_aux_f, "/indata/ac_aux_f", from_file);
ReadH5Dataset(m_indata.extcur, "/indata/extcur", from_file);
ReadH5Dataset(m_indata.aphi, "/indata/aphi", from_file);
ReadH5Dataset(m_indata.raxis_c, "/indata/raxis_c", from_file);
ReadH5Dataset(m_indata.zaxis_s, "/indata/zaxis_s", from_file);
if (m_indata.lasym) {
ReadH5Dataset(m_indata.raxis_s.emplace(), "/indata/raxis_s", from_file);
ReadH5Dataset(m_indata.zaxis_c.emplace(), "/indata/zaxis_c", from_file);
}

const int num_cols = 2 * m_indata.ntor + 1;

ReadH5Dataset(m_indata.rbc, "/indata/rbc", from_file);
if (m_indata.rbc.rows() != m_indata.mpol) {
return absl::InvalidArgumentError(
absl::StrCat("expected rbc to have ", m_indata.mpol, " rows, but had ",
m_indata.rbc.rows()));
}
if (m_indata.rbc.cols() != 2 * m_indata.ntor + 1) {
return absl::InvalidArgumentError(
absl::StrCat("expected rbc to have ", num_cols, " columns, but had ",
m_indata.rbc.cols()));
}

ReadH5Dataset(m_indata.zbs, "/indata/zbs", from_file);
if (m_indata.zbs.rows() != m_indata.mpol) {
return absl::InvalidArgumentError(
absl::StrCat("expected zbs to have ", m_indata.mpol, " rows, but had ",
m_indata.zbs.rows()));
}
if (m_indata.zbs.cols() != 2 * m_indata.ntor + 1) {
return absl::InvalidArgumentError(
absl::StrCat("expected zbs to have ", num_cols, " columns, but had ",
m_indata.zbs.cols()));
}

if (m_indata.lasym) {
ReadH5Dataset(m_indata.rbs.emplace(), "/indata/rbs", from_file);
if (m_indata.rbs->rows() != m_indata.mpol) {
return absl::InvalidArgumentError(
absl::StrCat("expected rbs to have ", m_indata.mpol,
" rows, but had ", m_indata.rbs->rows()));
}
if (m_indata.rbs->cols() != 2 * m_indata.ntor + 1) {
return absl::InvalidArgumentError(
absl::StrCat("expected rbs to have ", num_cols, " columns, but had ",
m_indata.rbs->cols()));
}

ReadH5Dataset(m_indata.zbc.emplace(), "/indata/zbc", from_file);
if (m_indata.zbc->rows() != m_indata.mpol) {
return absl::InvalidArgumentError(
absl::StrCat("expected zbc to have ", m_indata.mpol,
" rows, but had ", m_indata.zbc->rows()));
}
if (m_indata.zbc->cols() != 2 * m_indata.ntor + 1) {
return absl::InvalidArgumentError(
absl::StrCat("expected zbc to have ", num_cols, " columns, but had ",
m_indata.zbc->cols()));
}
}

return absl::OkStatus();
}

absl::StatusOr<VmecINDATA> VmecINDATA::FromJson(
const std::string& indata_json) {
absl::StatusOr<json> maybe_json = JsonParse(indata_json);
if (!maybe_json.ok()) {
return maybe_json.status();
}
const json& j = *maybe_json;

if (!j.is_object()) {
return absl::InvalidArgumentError("root JSON element is not an object");
}

VmecINDATA vmec_indata;

auto maybe_lasym = JsonReadBool(j, "lasym");
if (!maybe_lasym.ok()) {
return maybe_lasym.status();
}
if (maybe_lasym->has_value()) {
vmec_indata.lasym = maybe_lasym->value();
}

auto maybe_nfp = JsonReadInt(j, "nfp");
if (!maybe_nfp.ok()) {
return maybe_nfp.status();
}
if (maybe_nfp->has_value()) {
vmec_indata.nfp = maybe_nfp->value();
}

auto maybe_mpol = JsonReadInt(j, "mpol");
if (!maybe_mpol.ok()) {
return maybe_mpol.status();
}
if (maybe_mpol->has_value()) {
vmec_indata.mpol = maybe_mpol->value();
}

auto maybe_ntor = JsonReadInt(j, "ntor");
if (!maybe_ntor.ok()) {
return maybe_ntor.status();
}
if (maybe_ntor->has_value()) {
vmec_indata.ntor = maybe_ntor->value();
}

auto maybe_mpol_geometry = JsonReadInt(j, "mpol_geometry");
if (!maybe_mpol_geometry.ok()) {
return maybe_mpol_geometry.status();
}
if (maybe_mpol_geometry->has_value()) {
vmec_indata.mpol_geometry = maybe_mpol_geometry->value();
}

auto maybe_ntor_geometry = JsonReadInt(j, "ntor_geometry");
if (!maybe_ntor_geometry.ok()) {
return maybe_ntor_geometry.status();
}
if (maybe_ntor_geometry->has_value()) {
vmec_indata.ntor_geometry = maybe_ntor_geometry->value();
}

auto maybe_ntheta = JsonReadInt(j, "ntheta");
if (!maybe_ntheta.ok()) {
return maybe_ntheta.status();
}
if (maybe_ntheta->has_value()) {
vmec_indata.ntheta = maybe_ntheta->value();
}

auto maybe_nzeta = JsonReadInt(j, "nzeta");
if (!maybe_nzeta.ok()) {
return maybe_nzeta.status();
}
if (maybe_nzeta->has_value()) {
vmec_indata.nzeta = maybe_nzeta->value();
}

auto maybe_ns_array = JsonReadVectorInt(j, "ns_array");
if (!maybe_ns_array.ok()) {
return maybe_ns_array.status();
}
if (maybe_ns_array->has_value()) {
vmec_indata.ns_array = maybe_ns_array->value();
}

auto maybe_ftol_array = JsonReadVectorDouble(j, "ftol_array");
if (!maybe_ftol_array.ok()) {
return maybe_ftol_array.status();
}
if (maybe_ftol_array->has_value()) {
vmec_indata.ftol_array = maybe_ftol_array->value();
}

if (vmec_indata.ftol_array.size() != vmec_indata.ns_array.size()) {
return absl::InvalidArgumentError(absl::StrFormat(
"length of ftol_array (%ld) does not match length of ns_array (%ld)\n",
vmec_indata.ftol_array.size(), vmec_indata.ns_array.size()));
}

auto maybe_niter_array = JsonReadVectorInt(j, "niter_array");
if (!maybe_niter_array.ok()) {
return maybe_niter_array.status();
}
if (maybe_niter_array->has_value()) {
vmec_indata.niter_array = maybe_niter_array->value();
}

if (vmec_indata.niter_array.size() != vmec_indata.ns_array.size()) {
return absl::InvalidArgumentError(absl::StrFormat(
"length of niter_array (%ld) does not match length of ns_array (%ld)\n",
vmec_indata.niter_array.size(), vmec_indata.ns_array.size()));
}

auto maybe_phiedge = JsonReadDouble(j, "phiedge");
if (!maybe_phiedge.ok()) {
return maybe_phiedge.status();
}
if (maybe_phiedge->has_value()) {
vmec_indata.phiedge = maybe_phiedge->value();
}

auto maybe_ncurr = JsonReadInt(j, "ncurr");
if (!maybe_ncurr.ok()) {
return maybe_ncurr.status();
}
if (maybe_ncurr->has_value()) {
vmec_indata.ncurr = maybe_ncurr->value();
}

auto maybe_pmass_type = JsonReadString(j, "pmass_type");
if (!maybe_pmass_type.ok()) {
return maybe_pmass_type.status();
}
if (maybe_pmass_type->has_value()) {
vmec_indata.pmass_type = maybe_pmass_type->value();
}

auto maybe_am = JsonReadVectorDouble(j, "am");
if (!maybe_am.ok()) {
return maybe_am.status();
}
if (maybe_am->has_value()) {
vmec_indata.am = maybe_am->value();
}

auto maybe_am_aux_s = JsonReadVectorDouble(j, "am_aux_s");
if (!maybe_am_aux_s.ok()) {
return maybe_am_aux_s.status();
}
if (maybe_am_aux_s->has_value()) {
vmec_indata.am_aux_s = maybe_am_aux_s->value();
}

auto maybe_am_aux_f = JsonReadVectorDouble(j, "am_aux_f");
if (!maybe_am_aux_f.ok()) {
return maybe_am_aux_f.status();
}
if (maybe_am_aux_f->has_value()) {
vmec_indata.am_aux_f = maybe_am_aux_f->value();
}

if (vmec_indata.am_aux_f.size() != vmec_indata.am_aux_s.size()) {
return absl::InvalidArgumentError(absl::StrFormat(
"length of am_aux_f (%ld) does not match length of am_aux_s (%ld)\n",
vmec_indata.am_aux_f.size(), vmec_indata.am_aux_s.size()));
}

auto maybe_pres_scale = JsonReadDouble(j, "pres_scale");
if (!maybe_pres_scale.ok()) {
return maybe_pres_scale.status();
}
if (maybe_pres_scale->has_value()) {
vmec_indata.pres_scale = maybe_pres_scale->value();
}

auto maybe_gamma = JsonReadDouble(j, "gamma");
if (!maybe_gamma.ok()) {
return maybe_gamma.status();
}
if (maybe_gamma->has_value()) {
vmec_indata.gamma = maybe_gamma->value();
}

auto maybe_spres_ped = JsonReadDouble(j, "spres_ped");
if (!maybe_spres_ped.ok()) {
return maybe_spres_ped.status();
}
if (maybe_spres_ped->has_value()) {
vmec_indata.spres_ped = maybe_spres_ped->value();
}

auto maybe_piota_type = JsonReadString(j, "piota_type");
if (!maybe_piota_type.ok()) {
return maybe_piota_type.status();
}
if (maybe_piota_type->has_value()) {
vmec_indata.piota_type = maybe_piota_type->value();
}

auto maybe_ai = JsonReadVectorDouble(j, "ai");
if (!maybe_ai.ok()) {
return maybe_ai.status();
}
if (maybe_ai->has_value()) {
vmec_indata.ai = maybe_ai->value();
}

auto maybe_ai_aux_s = JsonReadVectorDouble(j, "ai_aux_s");
if (!maybe_ai_aux_s.ok()) {
return maybe_ai_aux_s.status();
}
if (maybe_ai_aux_s->has_value()) {
vmec_indata.ai_aux_s = maybe_ai_aux_s->value();
}

auto maybe_ai_aux_f = JsonReadVectorDouble(j, "ai_aux_f");
if (!maybe_ai_aux_f.ok()) {
return maybe_ai_aux_f.status();
}
if (maybe_ai_aux_f->has_value()) {
vmec_indata.ai_aux_f = maybe_ai_aux_f->value();
}

if (vmec_indata.ai_aux_f.size() != vmec_indata.ai_aux_s.size()) {
return absl::InvalidArgumentError(absl::StrFormat(
"length of ai_aux_f (%ld) does not match length of ai_aux_s (%ld)\n",
vmec_indata.ai_aux_f.size(), vmec_indata.ai_aux_s.size()));
}

auto maybe_pcurr_type = JsonReadString(j, "pcurr_type");
if (!maybe_pcurr_type.ok()) {
return maybe_pcurr_type.status();
}
if (maybe_pcurr_type->has_value()) {
vmec_indata.pcurr_type = maybe_pcurr_type->value();
}

auto maybe_ac = JsonReadVectorDouble(j, "ac");
if (!maybe_ac.ok()) {
return maybe_ac.status();
}
if (maybe_ac->has_value()) {
vmec_indata.ac = maybe_ac->value();
}

auto maybe_ac_aux_s = JsonReadVectorDouble(j, "ac_aux_s");
if (!maybe_ac_aux_s.ok()) {
return maybe_ac_aux_s.status();
}
if (maybe_ac_aux_s->has_value()) {
vmec_indata.ac_aux_s = maybe_ac_aux_s->value();
}

auto maybe_ac_aux_f = JsonReadVectorDouble(j, "ac_aux_f");
if (!maybe_ac_aux_f.ok()) {
return maybe_ac_aux_f.status();
}
if (maybe_ac_aux_f->has_value()) {
vmec_indata.ac_aux_f = maybe_ac_aux_f->value();
}

if (vmec_indata.ac_aux_f.size() != vmec_indata.ac_aux_s.size()) {
return absl::InvalidArgumentError(absl::StrFormat(
"length of ac_aux_f (%ld) does not match length of ac_aux_s (%ld)\n",
vmec_indata.ac_aux_f.size(), vmec_indata.ac_aux_s.size()));
}

auto maybe_curtor = JsonReadDouble(j, "curtor");
if (!maybe_curtor.ok()) {
return maybe_curtor.status();
}
if (maybe_curtor->has_value()) {
vmec_indata.curtor = maybe_curtor->value();
}

auto maybe_bloat = JsonReadDouble(j, "bloat");
if (!maybe_bloat.ok()) {
return maybe_bloat.status();
}
if (maybe_bloat->has_value()) {
vmec_indata.bloat = maybe_bloat->value();
}

auto maybe_lfreeb = JsonReadBool(j, "lfreeb");
if (!maybe_lfreeb.ok()) {
return maybe_lfreeb.status();
}
if (maybe_lfreeb->has_value()) {
vmec_indata.lfreeb = maybe_lfreeb->value();
}

auto maybe_mgrid_file = JsonReadString(j, "mgrid_file");
if (!maybe_mgrid_file.ok()) {
return maybe_mgrid_file.status();
}
if (maybe_mgrid_file->has_value()) {
vmec_indata.mgrid_file = maybe_mgrid_file->value();
}

auto maybe_extcur = JsonReadVectorDouble(j, "extcur");
if (!maybe_extcur.ok()) {
return maybe_extcur.status();
}
if (maybe_extcur->has_value()) {
vmec_indata.extcur = maybe_extcur->value();
}

auto maybe_nvacskip = JsonReadInt(j, "nvacskip");
if (!maybe_nvacskip.ok()) {
return maybe_nvacskip.status();
}
if (maybe_nvacskip->has_value()) {
vmec_indata.nvacskip = maybe_nvacskip->value();
}

auto maybe_signgs = JsonReadInt(j, "signgs");
if (!maybe_signgs.ok()) {
return maybe_signgs.status();
}
if (maybe_signgs->has_value()) {
vmec_indata.signgs = maybe_signgs->value();
}

auto maybe_free_boundary_method = JsonReadString(j, "free_boundary_method");
if (!maybe_free_boundary_method.ok()) {
return maybe_free_boundary_method.status();
}
if (maybe_free_boundary_method->has_value()) {
absl::StatusOr<FreeBoundaryMethod> status_or_free_boundary_method =
FreeBoundaryMethodFromString(maybe_free_boundary_method->value());
if (status_or_free_boundary_method.ok()) {
vmec_indata.free_boundary_method = status_or_free_boundary_method.value();
} else {
return status_or_free_boundary_method.status();
}
}

auto maybe_nstep = JsonReadInt(j, "nstep");
if (!maybe_nstep.ok()) {
return maybe_nstep.status();
}
if (maybe_nstep->has_value()) {
vmec_indata.nstep = maybe_nstep->value();
}

auto maybe_aphi = JsonReadVectorDouble(j, "aphi");
if (!maybe_aphi.ok()) {
return maybe_aphi.status();
}
if (maybe_aphi->has_value()) {
vmec_indata.aphi = maybe_aphi->value();
}

auto maybe_delt = JsonReadDouble(j, "delt");
if (!maybe_delt.ok()) {
return maybe_delt.status();
}
if (maybe_delt->has_value()) {
vmec_indata.delt = maybe_delt->value();
}

auto maybe_tcon0 = JsonReadDouble(j, "tcon0");
if (!maybe_tcon0.ok()) {
return maybe_tcon0.status();
}
if (maybe_tcon0->has_value()) {
vmec_indata.tcon0 = maybe_tcon0->value();
}

auto maybe_lforbal = JsonReadBool(j, "lforbal");
if (!maybe_lforbal.ok()) {
return maybe_lforbal.status();
}
if (maybe_lforbal->has_value()) {
vmec_indata.lforbal = maybe_lforbal->value();
}

auto maybe_iteration_style = JsonReadString(j, "iteration_style");
if (!maybe_iteration_style.ok()) {
return maybe_iteration_style.status();
}
if (maybe_iteration_style->has_value()) {
absl::StatusOr<IterationStyle> status_or_iteration_style =
IterationStyleFromString(maybe_iteration_style->value());
if (status_or_iteration_style.ok()) {
vmec_indata.iteration_style = status_or_iteration_style.value();
} else {
return status_or_iteration_style.status();
}
}

auto maybe_return_outputs_even_if_not_converged =
JsonReadBool(j, "return_outputs_even_if_not_converged");
if (!maybe_return_outputs_even_if_not_converged.ok()) {
return maybe_return_outputs_even_if_not_converged.status();
}
if (maybe_return_outputs_even_if_not_converged->has_value()) {
vmec_indata.return_outputs_even_if_not_converged =
maybe_return_outputs_even_if_not_converged->value();
}

const int expected_axis_size = vmec_indata.ntor + 1;
auto pad_or_reject_axis = [expected_axis_size](
Eigen::VectorXd& m_axis,
const char* name) -> absl::Status {
if (m_axis.size() > expected_axis_size) {
return absl::InvalidArgumentError(
absl::StrFormat("length of %s (%ld) exceeds ntor+1 (%d)\n", name,
m_axis.size(), expected_axis_size));
}
if (m_axis.size() < expected_axis_size) {
Eigen::VectorXd padded = Eigen::VectorXd::Zero(expected_axis_size);
padded.head(m_axis.size()) = m_axis;
m_axis = std::move(padded);
}
return absl::OkStatus();
};

auto maybe_raxis_c = JsonReadVectorDouble(j, "raxis_c");
if (!maybe_raxis_c.ok()) {
return maybe_raxis_c.status();
}
if (maybe_raxis_c->has_value()) {
vmec_indata.raxis_c = maybe_raxis_c->value();
}

if (absl::Status status = pad_or_reject_axis(vmec_indata.raxis_c, "raxis_c");
!status.ok()) {
return status;
}

auto maybe_zaxis_s = JsonReadVectorDouble(j, "zaxis_s");
if (!maybe_zaxis_s.ok()) {
return maybe_zaxis_s.status();
}
if (maybe_zaxis_s->has_value()) {
vmec_indata.zaxis_s = maybe_zaxis_s->value();
}

if (absl::Status status = pad_or_reject_axis(vmec_indata.zaxis_s, "zaxis_s");
!status.ok()) {
return status;
}

if (vmec_indata.lasym) {

vmec_indata.raxis_s.emplace().setZero(expected_axis_size);
vmec_indata.zaxis_c.emplace().setZero(expected_axis_size);

auto maybe_raxis_s = JsonReadVectorDouble(j, "raxis_s");
if (!maybe_raxis_s.ok()) {
return maybe_raxis_s.status();
}
if (maybe_raxis_s->has_value()) {
vmec_indata.raxis_s = maybe_raxis_s->value();
}

if (absl::Status status =
pad_or_reject_axis(vmec_indata.raxis_s.value(), "raxis_s");
!status.ok()) {
return status;
}

auto maybe_zaxis_c = JsonReadVectorDouble(j, "zaxis_c");
if (!maybe_zaxis_c.ok()) {
return maybe_zaxis_c.status();
}
if (maybe_zaxis_c->has_value()) {
vmec_indata.zaxis_c = maybe_zaxis_c->value();
}

if (absl::Status status =
pad_or_reject_axis(vmec_indata.zaxis_c.value(), "zaxis_c");
!status.ok()) {
return status;
}
}

auto maybe_rbc = BoundaryCoefficient::FromJson(j, "rbc");
if (!maybe_rbc.ok()) {
return maybe_rbc.status();
}
if (maybe_rbc->has_value()) {
vmec_indata.rbc.setZero(vmec_indata.mpol, 2 * vmec_indata.ntor + 1);
std::vector<BoundaryCoefficient> entries = maybe_rbc->value();
for (const BoundaryCoefficient& entry : entries) {
if (entry.m > vmec_indata.mpol - 1) {
LOG(INFO) << absl::StrFormat(
"Ignoring rbc entry with m = %d, since m is larger than (mpol - 1) "
"= %d",
entry.m, vmec_indata.mpol - 1);
continue;
}
if (std::abs(entry.n) > vmec_indata.ntor) {
LOG(INFO) << absl::StrFormat(
"Ignoring rbc entry with n = %d, since |n| is larger than ntor = "
"%d",
entry.n, vmec_indata.ntor);
continue;
}

const int index_along_n = vmec_indata.ntor + entry.n;

vmec_indata.rbc(entry.m, index_along_n) = entry.value;
}
}

auto maybe_zbs = BoundaryCoefficient::FromJson(j, "zbs");
if (!maybe_zbs.ok()) {
return maybe_zbs.status();
}
if (maybe_zbs->has_value()) {
vmec_indata.zbs.setZero(vmec_indata.mpol, 2 * vmec_indata.ntor + 1);
std::vector<BoundaryCoefficient> entries = maybe_zbs->value();
for (const BoundaryCoefficient& entry : entries) {
if (entry.m > vmec_indata.mpol - 1) {
LOG(INFO) << absl::StrFormat(
"Ignoring zbs entry with m = %d, since m is larger than (mpol - 1) "
"= %d",
entry.m, vmec_indata.mpol - 1);
continue;
}
if (std::abs(entry.n) > vmec_indata.ntor) {
LOG(INFO) << absl::StrFormat(
"Ignoring zbs entry with n = %d, since |n| is larger than ntor = "
"%d",
entry.n, vmec_indata.ntor);
continue;
}

const int index_along_n = vmec_indata.ntor + entry.n;

vmec_indata.zbs(entry.m, index_along_n) = entry.value;
}
}

if (vmec_indata.lasym) {

vmec_indata.rbs.emplace().setZero(vmec_indata.mpol,
2 * vmec_indata.ntor + 1);
vmec_indata.zbc.emplace().setZero(vmec_indata.mpol,
2 * vmec_indata.ntor + 1);

auto maybe_rbs = BoundaryCoefficient::FromJson(j, "rbs");
if (!maybe_rbs.ok()) {
return maybe_rbs.status();
}
if (maybe_rbs->has_value()) {
std::vector<BoundaryCoefficient> entries = maybe_rbs->value();
for (const BoundaryCoefficient& entry : entries) {
if (entry.m > vmec_indata.mpol - 1) {
LOG(INFO) << absl::StrFormat(
"Ignoring rbs entry with m = %d, since m is larger than (mpol - "
"1) = %d",
entry.m, vmec_indata.mpol - 1);
continue;
}
if (std::abs(entry.n) > vmec_indata.ntor) {
LOG(INFO) << absl::StrFormat(
"Ignoring rbs entry with n = %d, since |n| is larger than ntor = "
"%d",
entry.n, vmec_indata.ntor);
continue;
}

const int index_along_n = vmec_indata.ntor + entry.n;

(*vmec_indata.rbs)(entry.m, index_along_n) = entry.value;
}
}

auto maybe_zbc = BoundaryCoefficient::FromJson(j, "zbc");
if (!maybe_zbc.ok()) {
return maybe_zbc.status();
}
if (maybe_zbc->has_value()) {
std::vector<BoundaryCoefficient> entries = maybe_zbc->value();
for (const BoundaryCoefficient& entry : entries) {
if (entry.m > vmec_indata.mpol - 1) {
LOG(INFO) << absl::StrFormat(
"Ignoring zbc entry with m = %d, since m is larger than (mpol - "
"1) = %d",
entry.m, vmec_indata.mpol - 1);
continue;
}
if (std::abs(entry.n) > vmec_indata.ntor) {
LOG(INFO) << absl::StrFormat(
"Ignoring zbc entry with n = %d, since |n| is larger than ntor = "
"%d",
entry.n, vmec_indata.ntor);
continue;
}

const int index_along_n = vmec_indata.ntor + entry.n;

(*vmec_indata.zbc)(entry.m, index_along_n) = entry.value;
}
}
}

static constexpr bool kEnableInfoMessages = false;
absl::Status consistency_check_status =
IsConsistent(vmec_indata, kEnableInfoMessages);
if (!consistency_check_status.ok()) {
return consistency_check_status;
}

return vmec_indata;
}

VmecINDATA VmecINDATA::FromFile(
const std::filesystem::path& indata_json_file_path) {
absl::StatusOr<std::string> indata_json =
file_io::ReadFile(indata_json_file_path);

if (!indata_json.ok()) {
ErrorToException(
indata_json.status(),
"reading JSON file '" + indata_json_file_path.string() + "'");
}

absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
if (!vmec_indata.ok()) {
ErrorToException(vmec_indata.status(),
"creating VmecINDATA object from JSON (input file was '" +
indata_json_file_path.string() + "')");
}

return vmec_indata.value();
}

absl::StatusOr<std::string> VmecINDATA::ToJson() const {
nlohmann::json output;

output["lasym"] = lasym;
output["nfp"] = nfp;
output["mpol"] = mpol;
output["ntor"] = ntor;
output["mpol_geometry"] = mpol_geometry;
output["ntor_geometry"] = ntor_geometry;
output["ntheta"] = ntheta;
output["nzeta"] = nzeta;

output["ns_array"] = ns_array;
output["ftol_array"] = ftol_array;
output["niter_array"] = niter_array;

output["phiedge"] = phiedge;
output["ncurr"] = ncurr;

output["pmass_type"] = pmass_type;
output["am"] = am;
output["am_aux_s"] = am_aux_s;
output["am_aux_f"] = am_aux_f;
output["pres_scale"] = pres_scale;
output["gamma"] = gamma;
output["spres_ped"] = spres_ped;

output["piota_type"] = piota_type;
output["ai"] = ai;
output["ai_aux_s"] = ai_aux_s;
output["ai_aux_f"] = ai_aux_f;

output["pcurr_type"] = pcurr_type;
output["ac"] = ac;
output["ac_aux_s"] = ac_aux_s;
output["ac_aux_f"] = ac_aux_f;
output["curtor"] = curtor;
output["bloat"] = bloat;

output["lfreeb"] = lfreeb;
output["mgrid_file"] = mgrid_file;
output["extcur"] = extcur;
output["nvacskip"] = nvacskip;
output["signgs"] = signgs;
output["free_boundary_method"] = ToString(free_boundary_method);

output["nstep"] = nstep;
output["aphi"] = aphi;
output["delt"] = delt;
output["tcon0"] = tcon0;
output["lforbal"] = lforbal;
output["iteration_style"] = ToString(iteration_style);
output["return_outputs_even_if_not_converged"] =
return_outputs_even_if_not_converged;

output["raxis_c"] = raxis_c;
output["zaxis_s"] = zaxis_s;
if (lasym) {
output["raxis_s"] = *raxis_s;
output["zaxis_c"] = *zaxis_c;
}

output["rbc"] = std::vector<nlohmann::json>();
output["zbs"] = std::vector<nlohmann::json>();
if (lasym) {
output["rbs"] = std::vector<nlohmann::json>();
output["zbc"] = std::vector<nlohmann::json>();
}
nlohmann::json tmp_obj;
for (int m = 0; m < mpol; ++m) {
for (int n = -ntor; n <= ntor; ++n) {
tmp_obj["m"] = m;
tmp_obj["n"] = n;
tmp_obj["value"] = rbc(m, ntor + n);

auto push_nonzero = [&output, &tmp_obj](const std::string& key,
double value) {
tmp_obj["value"] = value;
if (tmp_obj["value"] != 0.0) {
output[key].push_back(tmp_obj);
}
};

push_nonzero("rbc", rbc(m, ntor + n));
push_nonzero("zbs", zbs(m, ntor + n));
if (lasym) {

push_nonzero("rbs", (*rbs)(m, ntor + n));
push_nonzero("zbc", (*zbc)(m, ntor + n));
}
}
}

return output.dump();
}

std::string VmecINDATA::ToJsonOrException() const {
const absl::StatusOr<std::string> json = ToJson();
if (!json.ok()) {
ErrorToException(json.status(), "converting VmecINDATA to JSON");
}
return *json;
}

absl::Status IsConsistent(const VmecINDATA& vmec_indata,
bool enable_info_messages) {

if (vmec_indata.nfp <= 0) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'nfp' needs to be > 0, but is %d", vmec_indata.nfp));
}

if (vmec_indata.mpol < 2) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'mpol' needs to be >= 2, but is %d\n",
vmec_indata.mpol));
}

if (vmec_indata.ntor < 0) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'ntor' needs to be >= 0, but is %d\n",
vmec_indata.ntor));
}

if (vmec_indata.ntheta < 0) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'ntheta' needs to be >= 0, but is %d\n",
vmec_indata.ntheta));
}

if (vmec_indata.nzeta < 0) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'nzeta' needs to be >= 0, but is %d\n",
vmec_indata.nzeta));
}

if (vmec_indata.signgs != -1 && vmec_indata.signgs != 1) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'signgs' needs to be -1 or +1, but is %d\n",
vmec_indata.signgs));
}

if (vmec_indata.nvacskip < 0) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'nvacskip' needs to be >= 0, but is %d\n",
vmec_indata.nvacskip));
}

const int NS_MIN = 3;

if (vmec_indata.ns_array.size() >
0) {
int largestNumSurfacesSoFar = -1;
for (Eigen::VectorXi::Index idx = 0; idx < vmec_indata.ns_array.size();
++idx) {
int ns = vmec_indata.ns_array[idx];
if (ns >= NS_MIN) {
if (ns < largestNumSurfacesSoFar) {

return absl::InvalidArgumentError(
absl::StrFormat("input variable 'ns_array' needs to increase "
"monotonically or stay constant, but entries %ld "
"and %ld are %d and %d, respectively",
idx - 1, idx, vmec_indata.ns_array[idx - 1], ns));
} else if (ns > largestNumSurfacesSoFar) {
largestNumSurfacesSoFar = ns;
}
} else {
return absl::InvalidArgumentError(
absl::StrFormat("values in input variable 'ns_array' need to be at "
"least %d, but value %d was found at index %ld\n",
NS_MIN, ns, idx));
}
}
} else {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'ns_array' needs to have at least one "
"entry, but size is %ld\n",
vmec_indata.ns_array.size()));
}

if (vmec_indata.ftol_array.size() < vmec_indata.ns_array.size()) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'ftol_array' needs an entry for every 'ns_array' "
"entry, but has %ld against %ld\n",
vmec_indata.ftol_array.size(), vmec_indata.ns_array.size()));
}
if (vmec_indata.niter_array.size() < vmec_indata.ns_array.size()) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'niter_array' needs an entry for every 'ns_array' "
"entry, but has %ld against %ld\n",
vmec_indata.niter_array.size(), vmec_indata.ns_array.size()));
}

for (Eigen::VectorXi::Index idx = 0; idx < vmec_indata.ns_array.size();
++idx) {
double ftol = vmec_indata.ftol_array[idx];
if (ftol < 0.0) {
return absl::InvalidArgumentError(
absl::StrFormat("values in input variable 'ftol_array' need to be "
"positive, but value %e was found at index %ld\n",
ftol, idx));
}
}

for (Eigen::VectorXi::Index idx = 0; idx < vmec_indata.ns_array.size();
++idx) {
int niter = vmec_indata.niter_array[idx];
if (niter <= 0) {
return absl::InvalidArgumentError(
absl::StrFormat("values in input variable 'niter_array' need to be "
"positive, but value %d was found at index %ld\n",
niter, idx));
}
}

if (vmec_indata.phiedge == 0.0) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'phiedge' must not be 0.0\n"));
}

if (vmec_indata.ncurr != 0 && vmec_indata.ncurr != 1) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'ncurr' must be 0 or 1, but is %d\n",
vmec_indata.ncurr));
}

if (absl::Status status = CheckProfile(
"pmass_type", vmec_indata.pmass_type, ProfileType::PRESSURE, "am",
vmec_indata.am_aux_s, vmec_indata.am_aux_f);
!status.ok()) {
return status;
}

if (vmec_indata.pres_scale < 0) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'pres_scale' must be positive, but was %g\n",
vmec_indata.pres_scale));
}

if (vmec_indata.gamma == 1.0) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'gamma' must not be 1.0\n"));
}

if (vmec_indata.spres_ped <= 0.0 || vmec_indata.spres_ped > 1) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'spres_ped' must be > 0 and <= 1, but is %g\n",
vmec_indata.spres_ped));
}

if (absl::Status status =
CheckProfile("piota_type", vmec_indata.piota_type, ProfileType::IOTA,
"ai", vmec_indata.ai_aux_s, vmec_indata.ai_aux_f);
!status.ok()) {
return status;
}

if (absl::Status status = CheckProfile(
"pcurr_type", vmec_indata.pcurr_type, ProfileType::CURRENT, "ac",
vmec_indata.ac_aux_s, vmec_indata.ac_aux_f);
!status.ok()) {
return status;
}

if (vmec_indata.ncurr == 0) {
if (vmec_indata.bloat != 1.0) {

return absl::InvalidArgumentError(absl::StrFormat(
"'bloat' must be 1.0 for ncurr == 0 (constrained-iota), but is %g\n",
vmec_indata.bloat));
}
}

if (vmec_indata.lfreeb) {

if (vmec_indata.nvacskip < 1) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable nvacskip needs to be > 0, but is %d\n",
vmec_indata.nvacskip));
}

if (vmec_indata.free_boundary_method != FreeBoundaryMethod::NESTOR &&
vmec_indata.free_boundary_method != FreeBoundaryMethod::ONLY_COILS) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'free_boundary_method' must be "
"'nestor' or 'only_coils', but is %s\n",
ToString(vmec_indata.free_boundary_method)));
}

if (vmec_indata.free_boundary_method == FreeBoundaryMethod::ONLY_COILS &&
(vmec_indata.curtor != 0.0 || vmec_indata.pres_scale != 0.0)) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variables 'curtor' and 'pres_scale' must be zero when "
"'free_boundary_method' is 'only_coils', but are %g and %g\n",
vmec_indata.curtor, vmec_indata.pres_scale));
}
}

if (vmec_indata.nstep <= 0) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'nstep' must be > 0, but is %d\n", vmec_indata.nstep));
}

if (vmec_indata.delt <= 0.0 || vmec_indata.delt > 10.0) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'delt' has to be in the range ]0.0, 10.0], but is %g\n",
vmec_indata.delt));
}

if (vmec_indata.tcon0 < 0.0 || vmec_indata.tcon0 > 1.0) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'tcon0' has to be in the range [0.0, 1.0], but is %g\n",
vmec_indata.tcon0));
}

if (vmec_indata.iteration_style != IterationStyle::VMEC_8_52 &&
vmec_indata.iteration_style != IterationStyle::PARVMEC) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'iteration_style' must be 'vmec_8_52' "
"or 'parvmec', but "
"is %s\n",
ToString(vmec_indata.iteration_style)));
}

const int expected_axis_size = vmec_indata.ntor + 1;

if (vmec_indata.raxis_c.size() != expected_axis_size) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'raxis_c' has wrong size: should be %i, but it is %i.",
expected_axis_size, vmec_indata.raxis_c.size()));
}

if (vmec_indata.zaxis_s.size() != expected_axis_size) {
return absl::InvalidArgumentError(absl::StrFormat(
"input variable 'zaxis_s' has wrong size: should be %i, but it is %i.",
expected_axis_size, vmec_indata.zaxis_s.size()));
}

if (vmec_indata.lasym) {

if (!vmec_indata.raxis_s.has_value()) {
return absl::InvalidArgumentError(
"input variable 'raxis_s' has to be set when 'lasym' is true.");
}
if (!vmec_indata.zaxis_c.has_value()) {
return absl::InvalidArgumentError(
"input variable 'zaxis_c' has to be set when 'lasym' is true.");
}

if (vmec_indata.raxis_s->size() != expected_axis_size) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'raxis_s' has wrong size: should be "
"%i, but it is %i.",
expected_axis_size, vmec_indata.raxis_s->size()));
}

if (vmec_indata.zaxis_c->size() != expected_axis_size) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'zaxis_c' has wrong size: should be "
"%i, but it is %i.",
expected_axis_size, vmec_indata.zaxis_c->size()));
}
} else {

if (vmec_indata.raxis_s.has_value()) {
return absl::InvalidArgumentError(
"input variable 'raxis_s' should not be set when 'lasym' is false.");
}
if (vmec_indata.zaxis_c.has_value()) {
return absl::InvalidArgumentError(
"input variable 'zaxis_c' should not be set when 'lasym' is false.");
}
}

if (vmec_indata.rbc.rows() != vmec_indata.mpol) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'rbc' has wrong number of rows: should "
"be %i, but it is %i.",
vmec_indata.mpol, vmec_indata.rbc.rows()));
}
if (vmec_indata.rbc.cols() != 2 * vmec_indata.ntor + 1) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'rbc' has wrong number of columns: "
"should be %i, but it is %i.",
2 * vmec_indata.ntor + 1, vmec_indata.rbc.cols()));
}

if (vmec_indata.zbs.rows() != vmec_indata.mpol) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'zbs' has wrong number of rows: should "
"be %i, but it is %i.",
vmec_indata.mpol, vmec_indata.zbs.rows()));
}
if (vmec_indata.zbs.cols() != 2 * vmec_indata.ntor + 1) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'zbs' has wrong number of columns: "
"should be %i, but it is %i.",
2 * vmec_indata.ntor + 1, vmec_indata.zbs.cols()));
}

if (vmec_indata.lasym) {

if (!vmec_indata.rbs.has_value()) {
return absl::InvalidArgumentError(
"input variable 'rbs' has to be set when 'lasym' is true.");
}
if (!vmec_indata.zbc.has_value()) {
return absl::InvalidArgumentError(
"input variable 'zbc' has to be set when 'lasym' is true.");
}

if (vmec_indata.rbs->rows() != vmec_indata.mpol) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'rbs' has wrong number of rows: "
"should be %i, but it is %i.",
vmec_indata.mpol, vmec_indata.rbs->rows()));
}
if (vmec_indata.rbs->cols() != 2 * vmec_indata.ntor + 1) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'rbs' has wrong number of columns: "
"should be %i, but it is %i.",
2 * vmec_indata.ntor + 1, vmec_indata.rbs->cols()));
}

if (vmec_indata.zbc->rows() != vmec_indata.mpol) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'zbc' has wrong number of rows: "
"should be %i, but it is %i.",
vmec_indata.mpol, vmec_indata.zbc->rows()));
}
if (vmec_indata.zbc->cols() != 2 * vmec_indata.ntor + 1) {
return absl::InvalidArgumentError(
absl::StrFormat("input variable 'zbc' has wrong number of columns: "
"should be %i, but it is %i.",
2 * vmec_indata.ntor + 1, vmec_indata.zbc->cols()));
}
} else {
if (vmec_indata.rbs.has_value()) {
return absl::InvalidArgumentError(
"input variable 'rbs' should not be set when 'lasym' is false.");
}
if (vmec_indata.zbc.has_value()) {
return absl::InvalidArgumentError(
"input variable 'zbc' should not be set when 'lasym' is false.");
}
}

if (enable_info_messages && vmec_indata.zaxis_s[0] != 0.0) {

LOG(INFO) << absl::StrFormat(
"ignoring irrelevant zaxis_s entry for n=0: %g\n",
vmec_indata.zaxis_s[0]);
}

if (enable_info_messages && vmec_indata.zbs(0, vmec_indata.ntor) != 0.0) {

LOG(INFO) << absl::StrFormat(
"ignoring irrelevant zbs entry for m=0, n=0: %g\n",
vmec_indata.zbs(0, vmec_indata.ntor));
}

if (vmec_indata.lasym) {

if (enable_info_messages && (*vmec_indata.raxis_s)[0] != 0.0) {

LOG(INFO) << absl::StrFormat(
"ignoring irrelevant raxis_s entry for n=0: %g\n",
(*vmec_indata.raxis_s)[0]);
}

if (enable_info_messages &&
(*vmec_indata.rbs)(0, vmec_indata.ntor) != 0.0) {

LOG(INFO) << absl::StrFormat(
"ignoring irrelevant rbs entry for m=0, n=0: %g\n",
(*vmec_indata.rbs)(0, vmec_indata.ntor));
}
}

return absl::OkStatus();
}

VmecINDATA VmecINDATA::Copy() const { return *this; }

}

// source: vmecpp/free_boundary/external_magnetic_field/external_magnetic_field.cc

// header: vmecpp/free_boundary/external_magnetic_field/external_magnetic_field.h

#ifndef VMECPP_FREE_BOUNDARY_EXTERNAL_MAGNETIC_FIELD_EXTERNAL_MAGNETIC_FIELD_H_
#define VMECPP_FREE_BOUNDARY_EXTERNAL_MAGNETIC_FIELD_EXTERNAL_MAGNETIC_FIELD_H_

#include <Eigen/Dense>
#include <span>

#include "absl/status/status.h"

// header: vmecpp/free_boundary/mgrid_provider/mgrid_provider.h

#ifndef VMECPP_FREE_BOUNDARY_MGRID_PROVIDER_MGRID_PROVIDER_H_
#define VMECPP_FREE_BOUNDARY_MGRID_PROVIDER_MGRID_PROVIDER_H_

#include <Eigen/Dense>
#include <filesystem>
#include <string>
#include <vector>

#include "absl/status/status.h"

namespace vmecpp {

class MGridProvider {
public:
MGridProvider();

absl::Status LoadFile(const std::filesystem::path& filename,
const Eigen::VectorXd& coil_currents);

absl::Status LoadFields(
const makegrid::MagneticFieldResponseTable& magnetic_response_table,
const Eigen::VectorXd& coil_currents);

void SetFixedMagneticField(const Eigen::VectorXd& fixed_br,
const Eigen::VectorXd& fixed_bp,
const Eigen::VectorXd& fixed_bz);

[[nodiscard]] absl::Status interpolate(int ztMin, int ztMax, int nZeta,
int nZnT, const Eigen::VectorXd& r,
const Eigen::VectorXd& z,
Eigen::VectorXd& m_interpBr,
Eigen::VectorXd& m_interpBp,
Eigen::VectorXd& m_interpBz) const;

Eigen::VectorXd bR;
Eigen::VectorXd bP;
Eigen::VectorXd bZ;

int nfp;

int numR;
double minR;
double maxR;
double deltaR;

int numZ;
double minZ;
double maxZ;
double deltaZ;

int numPhi;

int nextcur;

std::string mgrid_mode;

std::vector<std::string> coil_group_names;

bool IsLoaded() const { return has_mgrid_loaded_; }

private:

void ResetAccumulatedField();

template <typename ContributionAt>
void AccumulateCircuit(double coil_current, ContributionAt contribution_at) {
const int num_grid_points = numPhi * numZ * numR;
for (int linear_index = 0; linear_index < num_grid_points; ++linear_index) {
const auto [b_r, b_p, b_z] = contribution_at(linear_index);
bR[linear_index] += b_r * coil_current;
bP[linear_index] += b_p * coil_current;
bZ[linear_index] += b_z * coil_current;
}
}

bool has_mgrid_loaded_;
bool has_fixed_field_;

Eigen::VectorXd fixed_br_;
Eigen::VectorXd fixed_bp_;
Eigen::VectorXd fixed_bz_;
};

}

#endif

// header: vmecpp/free_boundary/surface_geometry/surface_geometry.h

#ifndef VMECPP_FREE_BOUNDARY_SURFACE_GEOMETRY_SURFACE_GEOMETRY_H_
#define VMECPP_FREE_BOUNDARY_SURFACE_GEOMETRY_SURFACE_GEOMETRY_H_

#include <Eigen/Dense>
#include <span>

// header: vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h

#ifndef VMECPP_COMMON_FOURIER_BASIS_FAST_TOROIDAL_FOURIER_BASIS_FAST_TOROIDAL_H_
#define VMECPP_COMMON_FOURIER_BASIS_FAST_TOROIDAL_FOURIER_BASIS_FAST_TOROIDAL_H_

#endif

// header: vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h

#ifndef VMECPP_FREE_BOUNDARY_TANGENTIAL_PARTITIONING_TANGENTIAL_PARTITIONING_H_
#define VMECPP_FREE_BOUNDARY_TANGENTIAL_PARTITIONING_TANGENTIAL_PARTITIONING_H_

namespace vmecpp {

class TangentialPartitioning {
public:
explicit TangentialPartitioning(int nZnT, int num_threads = 1,
int thread_id = 0);

void adjustPartitioning(int nZnT);
int get_thread_id() const;
int get_num_threads() const;

int ztMin;
int ztMax;

private:
int num_threads_;
int thread_id_;
};

}

#endif

namespace vmecpp {

class SurfaceGeometry {
public:
SurfaceGeometry(const Sizes* s, const FourierBasisFastToroidal* fb,
const TangentialPartitioning* tp);

void update(
const std::span<const double> rCC, const std::span<const double> rSS,
const std::span<const double> rSC, const std::span<const double> rCS,
const std::span<const double> zSC, const std::span<const double> zCS,
const std::span<const double> zCC, const std::span<const double> zSS,
int signOfJacobian, bool fullUpdate);

Eigen::VectorXd cos_per;

Eigen::VectorXd sin_per;

Eigen::VectorXd cos_phi;

Eigen::VectorXd sin_phi;

Eigen::VectorXd r1b;

Eigen::VectorXd rub;

Eigen::VectorXd rvb;

Eigen::VectorXd z1b;

Eigen::VectorXd zub;

Eigen::VectorXd zvb;

Eigen::VectorXd ruu;

Eigen::VectorXd ruv;

Eigen::VectorXd rvv;

Eigen::VectorXd zuu;

Eigen::VectorXd zuv;

Eigen::VectorXd zvv;

Eigen::VectorXd snr;

Eigen::VectorXd snv;

Eigen::VectorXd snz;

Eigen::VectorXd guu;

Eigen::VectorXd guv;

Eigen::VectorXd gvv;

Eigen::VectorXd auu;

Eigen::VectorXd auv;

Eigen::VectorXd avv;

Eigen::VectorXd drv;

Eigen::VectorXd rzb2;

Eigen::VectorXd rcosuv;

Eigen::VectorXd rsinuv;

Eigen::VectorXd r1b_asym;
Eigen::VectorXd z1b_asym;
Eigen::VectorXd rub_asym;
Eigen::VectorXd rvb_asym;
Eigen::VectorXd zub_asym;
Eigen::VectorXd zvb_asym;
Eigen::VectorXd ruu_asym;
Eigen::VectorXd ruv_asym;
Eigen::VectorXd rvv_asym;
Eigen::VectorXd zuu_asym;
Eigen::VectorXd zuv_asym;
Eigen::VectorXd zvv_asym;

private:
const Sizes& s_;
const FourierBasisFastToroidal& fb_;
const TangentialPartitioning& tp_;

void computeConstants();

void inverseDFT(const std::span<const double> rCC,
const std::span<const double> rSS,
const std::span<const double> rSC,
const std::span<const double> rCS,
const std::span<const double> zSC,
const std::span<const double> zCS,
const std::span<const double> zCC,
const std::span<const double> zSS, bool fullUpdate);

void derivedSurfaceQuantities(int signOfJacobian, bool fullUpdate);
};

}

#endif

namespace vmecpp {

class ExternalMagneticField {
public:
ExternalMagneticField(const Sizes* s, const TangentialPartitioning* tp,
const SurfaceGeometry* sg, const MGridProvider* mgrid);

[[nodiscard]] absl::Status update(const std::span<const double> rAxis,
const std::span<const double> zAxis,
double netToroidalCurrent);

Eigen::VectorXd axisXYZ;
Eigen::VectorXd surfaceXYZ;
Eigen::VectorXd bCoilsXYZ;

Eigen::VectorXd interpBr;
Eigen::VectorXd interpBp;
Eigen::VectorXd interpBz;

double axis_current;
Eigen::VectorXd curtorBr;
Eigen::VectorXd curtorBp;
Eigen::VectorXd curtorBz;

Eigen::VectorXd bSubU;
Eigen::VectorXd bSubV;
Eigen::VectorXd bDotN;

private:

static constexpr bool kUseAbscabForAxisCurrent = false;

static constexpr int kAxisymmetricToroidalReplication = 64;
int nvper_;

const Sizes& s_;
const TangentialPartitioning& tp_;

const SurfaceGeometry& sg_;
const MGridProvider& mgrid_;

void updateInterpolation();

void AddAxisCurrentFieldAbscab(const std::span<const double> rAxis,
const std::span<const double> zAxis,
double netToroidalCurrent);

void AddAxisCurrentFieldSimple(const std::span<const double> rAxis,
const std::span<const double> zAxis,
double netToroidalCurrent);

void covariantAndNormalComponents();
};

}

#endif

#include <cmath>

#include "absl/algorithm/container.h"

namespace vmecpp {

ExternalMagneticField::ExternalMagneticField(const Sizes* s,
const TangentialPartitioning* tp,
const SurfaceGeometry* sg,
const MGridProvider* mgrid)
: s_(*s), tp_(*tp), sg_(*sg), mgrid_(*mgrid) {

nvper_ = (s_.nZeta == 1) ? kAxisymmetricToroidalReplication : s_.nfp;

axisXYZ.resize(3 * (s_.nZeta * nvper_ + 1));

const int numLocal = tp_.ztMax - tp_.ztMin;

surfaceXYZ.resize(3 * numLocal);
bCoilsXYZ.resize(3 * numLocal);

interpBr.resize(numLocal);
interpBp.resize(numLocal);
interpBz.resize(numLocal);

curtorBr.resize(numLocal);
curtorBp.resize(numLocal);
curtorBz.resize(numLocal);

bSubU.resize(numLocal);
bSubV.resize(numLocal);
bDotN.resize(numLocal);
}

absl::Status ExternalMagneticField::update(const std::span<const double> rAxis,
const std::span<const double> zAxis,
double netToroidalCurrent) {
#ifdef _OPENMP
#pragma omp barrier
#endif

absl::Status interpolation_status =
mgrid_.interpolate(tp_.ztMin, tp_.ztMax, s_.nZeta, s_.nZnT, sg_.r1b,
sg_.z1b, interpBr, interpBp, interpBz);

#ifdef _OPENMP
#pragma omp barrier
#endif

if (kUseAbscabForAxisCurrent) {
AddAxisCurrentFieldAbscab(rAxis, zAxis, netToroidalCurrent);
} else {
AddAxisCurrentFieldSimple(rAxis, zAxis, netToroidalCurrent);
}

#ifdef _OPENMP
#pragma omp barrier
#endif

covariantAndNormalComponents();

#ifdef _OPENMP
#pragma omp barrier
#endif

return interpolation_status;
}

void ExternalMagneticField::AddAxisCurrentFieldAbscab(
const std::span<const double> rAxis, const std::span<const double> zAxis,
double netToroidalCurrent) {

for (int k = 0; k < s_.nZeta; ++k) {
axisXYZ[k * 3 + 0] = rAxis[k] * sg_.cos_phi[k];
axisXYZ[k * 3 + 1] = rAxis[k] * sg_.sin_phi[k];
axisXYZ[k * 3 + 2] = zAxis[k];
}

const double omega_per = 2.0 * M_PI / nvper_;
for (int p = 1; p < nvper_; ++p) {
const double cos_per = std::cos(omega_per * p);
const double sin_per = std::sin(omega_per * p);
for (int k = 0; k < s_.nZeta; ++k) {
axisXYZ[(p * s_.nZeta + k) * 3 + 0] =
cos_per * axisXYZ[k * 3 + 0] - sin_per * axisXYZ[k * 3 + 1];
axisXYZ[(p * s_.nZeta + k) * 3 + 1] =
sin_per * axisXYZ[k * 3 + 0] + cos_per * axisXYZ[k * 3 + 1];
axisXYZ[(p * s_.nZeta + k) * 3 + 2] = zAxis[k];
}
}

axisXYZ[s_.nZeta * nvper_ * 3 + 0] = axisXYZ[0];
axisXYZ[s_.nZeta * nvper_ * 3 + 1] = axisXYZ[1];
axisXYZ[s_.nZeta * nvper_ * 3 + 2] = axisXYZ[2];

for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
const int k = kl % s_.nZeta;
surfaceXYZ[(kl - tp_.ztMin) * 3 + 0] = sg_.r1b[kl] * sg_.cos_phi[k];
surfaceXYZ[(kl - tp_.ztMin) * 3 + 1] = sg_.r1b[kl] * sg_.sin_phi[k];
surfaceXYZ[(kl - tp_.ztMin) * 3 + 2] = sg_.z1b[kl];
}

axis_current = netToroidalCurrent;

bCoilsXYZ.setZero();

int numProcessors = 1;
abscab::magneticFieldPolygonFilament(s_.nZeta * nvper_ + 1, axisXYZ.data(),
axis_current, tp_.ztMax - tp_.ztMin,
surfaceXYZ.data(), bCoilsXYZ.data(),
numProcessors);

for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
int k = kl % s_.nZeta;

double _bX = bCoilsXYZ[3 * (kl - tp_.ztMin) + 0];
double _bY = bCoilsXYZ[3 * (kl - tp_.ztMin) + 1];
double _bZ = bCoilsXYZ[3 * (kl - tp_.ztMin) + 2];

double _bR = sg_.cos_phi[k] * _bX + sg_.sin_phi[k] * _bY;
double _bP = sg_.cos_phi[k] * _bY - sg_.sin_phi[k] * _bX;

curtorBr[kl - tp_.ztMin] = _bR;
curtorBp[kl - tp_.ztMin] = _bP;
curtorBz[kl - tp_.ztMin] = _bZ;
}
}

void ExternalMagneticField::AddAxisCurrentFieldSimple(
const std::span<const double> rAxis, const std::span<const double> zAxis,
double netToroidalCurrent) {

for (int k = 0; k < s_.nZeta; ++k) {
axisXYZ[k * 3 + 0] = rAxis[k] * sg_.cos_phi[k];
axisXYZ[k * 3 + 1] = rAxis[k] * sg_.sin_phi[k];
axisXYZ[k * 3 + 2] = zAxis[k];
}

const double omega_per = 2.0 * M_PI / nvper_;
for (int p = 1; p < nvper_; ++p) {
const double cos_per = std::cos(omega_per * p);
const double sin_per = std::sin(omega_per * p);
for (int k = 0; k < s_.nZeta; ++k) {
axisXYZ[(p * s_.nZeta + k) * 3 + 0] =
cos_per * axisXYZ[k * 3 + 0] - sin_per * axisXYZ[k * 3 + 1];
axisXYZ[(p * s_.nZeta + k) * 3 + 1] =
sin_per * axisXYZ[k * 3 + 0] + cos_per * axisXYZ[k * 3 + 1];
axisXYZ[(p * s_.nZeta + k) * 3 + 2] = zAxis[k];
}
}

axisXYZ[s_.nZeta * nvper_ * 3 + 0] = axisXYZ[0];
axisXYZ[s_.nZeta * nvper_ * 3 + 1] = axisXYZ[1];
axisXYZ[s_.nZeta * nvper_ * 3 + 2] = axisXYZ[2];

for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
const int k = kl % s_.nZeta;
surfaceXYZ[(kl - tp_.ztMin) * 3 + 0] = sg_.r1b[kl] * sg_.cos_phi[k];
surfaceXYZ[(kl - tp_.ztMin) * 3 + 1] = sg_.r1b[kl] * sg_.sin_phi[k];
surfaceXYZ[(kl - tp_.ztMin) * 3 + 2] = sg_.z1b[kl];
}

axis_current = netToroidalCurrent;

bCoilsXYZ.setZero();

const double magnetic_field_scale = 1.0e-7 * netToroidalCurrent * 2.0;

for (int source_index = 0; source_index < s_.nZeta * nvper_; ++source_index) {
const double segment_dx =
axisXYZ[(source_index + 1) * 3 + 0] - axisXYZ[source_index * 3 + 0];
const double segment_dy =
axisXYZ[(source_index + 1) * 3 + 1] - axisXYZ[source_index * 3 + 1];
const double segment_dz =
axisXYZ[(source_index + 1) * 3 + 2] - axisXYZ[source_index * 3 + 2];

const double segment_length =
std::sqrt(segment_dx * segment_dx + segment_dy * segment_dy +
segment_dz * segment_dz);

for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
const int kl_local = kl - tp_.ztMin;

const double r_i_x =
surfaceXYZ[kl_local * 3 + 0] - axisXYZ[source_index * 3 + 0];
const double r_i_y =
surfaceXYZ[kl_local * 3 + 1] - axisXYZ[source_index * 3 + 1];
const double r_i_z =
surfaceXYZ[kl_local * 3 + 2] - axisXYZ[source_index * 3 + 2];
const double r_i =
std::sqrt(r_i_x * r_i_x + r_i_y * r_i_y + r_i_z * r_i_z);

const double r_f_x =
surfaceXYZ[kl_local * 3 + 0] - axisXYZ[(source_index + 1) * 3 + 0];
const double r_f_y =
surfaceXYZ[kl_local * 3 + 1] - axisXYZ[(source_index + 1) * 3 + 1];
const double r_f_z =
surfaceXYZ[kl_local * 3 + 2] - axisXYZ[(source_index + 1) * 3 + 2];
const double r_f =
std::sqrt(r_f_x * r_f_x + r_f_y * r_f_y + r_f_z * r_f_z);

const double r_i_plus_r_f = r_i + r_f;

const double magnetic_field_magnitude =
magnetic_field_scale * r_i_plus_r_f /
(r_i * r_f *
(r_i_plus_r_f * r_i_plus_r_f - segment_length * segment_length));

bCoilsXYZ[kl_local * 3 + 0] +=
magnetic_field_magnitude * (segment_dy * r_i_z - segment_dz * r_i_y);
bCoilsXYZ[kl_local * 3 + 1] +=
magnetic_field_magnitude * (segment_dz * r_i_x - segment_dx * r_i_z);
bCoilsXYZ[kl_local * 3 + 2] +=
magnetic_field_magnitude * (segment_dx * r_i_y - segment_dy * r_i_x);
}
}

for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
int k = kl % s_.nZeta;

double _bX = bCoilsXYZ[3 * (kl - tp_.ztMin) + 0];
double _bY = bCoilsXYZ[3 * (kl - tp_.ztMin) + 1];
double _bZ = bCoilsXYZ[3 * (kl - tp_.ztMin) + 2];

double _bR = sg_.cos_phi[k] * _bX + sg_.sin_phi[k] * _bY;
double _bP = sg_.cos_phi[k] * _bY - sg_.sin_phi[k] * _bX;

curtorBr[kl - tp_.ztMin] = _bR;
curtorBp[kl - tp_.ztMin] = _bP;
curtorBz[kl - tp_.ztMin] = _bZ;
}
}

void ExternalMagneticField::covariantAndNormalComponents() {

const int derivOffset = s_.lasym ? 0 : tp_.ztMin;
for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {

double fullBr = interpBr[kl - tp_.ztMin] + curtorBr[kl - tp_.ztMin];
double fullBp = interpBp[kl - tp_.ztMin] + curtorBp[kl - tp_.ztMin];
double fullBz = interpBz[kl - tp_.ztMin] + curtorBz[kl - tp_.ztMin];

bSubU[kl - tp_.ztMin] =
fullBr * sg_.rub[kl - derivOffset] + fullBz * sg_.zub[kl - derivOffset];
bSubV[kl - tp_.ztMin] = fullBr * sg_.rvb[kl - derivOffset] +
fullBz * sg_.zvb[kl - derivOffset] +
fullBp * sg_.r1b[kl];

bDotN[kl - tp_.ztMin] =
-(fullBr * sg_.snr[kl - tp_.ztMin] + fullBp * sg_.snv[kl - tp_.ztMin] +
fullBz * sg_.snz[kl - tp_.ztMin]);
}
}

}

// source: vmecpp/free_boundary/laplace_solver/laplace_solver.cc

// header: vmecpp/free_boundary/laplace_solver/laplace_solver.h

#ifndef VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_
#define VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_

#include <Eigen/Dense>
#include <vector>

namespace vmecpp {

class LaplaceSolver {
public:

LaplaceSolver(const Sizes* s, const FourierBasisFastToroidal* fb,
const TangentialPartitioning* tp, int nf, int mf,
std::span<double> matrixShare,
Eigen::PartialPivLU<Eigen::MatrixXd>* lu_decomposition,
std::span<double> bvecShare, std::span<double> reduce_slots);

void TransformGreensFunctionDerivative(const Eigen::VectorXd& greenp);
void SymmetriseSourceTerm(const Eigen::VectorXd& gstore);
void AccumulateFullGrpmn(const Eigen::VectorXd& grpmn_sin_singular,
const Eigen::VectorXd& grpmn_cos_singular);
void PerformToroidalFourierTransforms();
void PerformPoloidalFourierTransforms();

void BuildMatrix();
void DecomposeMatrix();

void SolveForPotential(const Eigen::VectorXd& bvec_sin_singular,
const Eigen::VectorXd& bvec_cos_singular = {});

Eigen::VectorXd grpmn_sin;

Eigen::VectorXd grpmn_cos;

Eigen::VectorXd gstore_symm;

Eigen::VectorXd gstore_asym;

Eigen::VectorXd bcos;
Eigen::VectorXd bsin;

Eigen::VectorXd bcos_asym;
Eigen::VectorXd bsin_asym;

Eigen::VectorXd actemp;
Eigen::VectorXd astemp;

Eigen::VectorXd actemp_cos;
Eigen::VectorXd astemp_cos;

Eigen::VectorXd bvec_sin;
Eigen::VectorXd bvec_cos;
Eigen::VectorXd amat_sin_sin;
Eigen::VectorXd amat_sin_cos;
Eigen::VectorXd amat_cos_sin;
Eigen::VectorXd amat_cos_cos;

private:
const Sizes& s_;
const FourierBasisFastToroidal& fb_;
const TangentialPartitioning& tp_;

int nf;
int mf;

std::span<double> matrixShare;

Eigen::PartialPivLU<Eigen::MatrixXd>* lu_decomposition_;
std::span<double> bvecShare;

std::span<double> reduce_slots_;

int numLocal;

Eigen::MatrixXd cosnv_scaled;
Eigen::MatrixXd sinnv_scaled;

Eigen::MatrixXd cosmui_scaled;
Eigen::MatrixXd sinmui_scaled;

Eigen::VectorXd grpOdd;
Eigen::VectorXd grpEvn;
};

}

#endif

#include <algorithm>
#include <cmath>
#include <vector>

#include "absl/log/check.h"

namespace vmecpp {

LaplaceSolver::LaplaceSolver(
const Sizes* s, const FourierBasisFastToroidal* fb,
const TangentialPartitioning* tp, int nf, int mf,
std::span<double> matrixShare,
Eigen::PartialPivLU<Eigen::MatrixXd>* lu_decomposition,
std::span<double> bvecShare, std::span<double> reduce_slots)
: s_(*s),
fb_(*fb),
tp_(*tp),
nf(nf),
mf(mf),
matrixShare(matrixShare),
lu_decomposition_(lu_decomposition),
bvecShare(bvecShare),
reduce_slots_(reduce_slots) {

numLocal = tp_.ztMax - tp_.ztMin;

grpOdd.resize(s_.nThetaReduced * s_.nZeta);
grpOdd.setZero();
if (s_.lasym) {
grpEvn.resize(s_.nThetaReduced * s_.nZeta);
grpEvn.setZero();
}

const int mnpd = (2 * nf + 1) * (mf + 1);
grpmn_sin.resize(mnpd * numLocal);
grpmn_sin.setZero();
if (s_.lasym) {
grpmn_cos.resize(mnpd * numLocal);
grpmn_cos.setZero();
}

gstore_symm.resize(s_.nThetaReduced * s_.nZeta);
gstore_symm.setZero();

const int size_b = s_.nThetaReduced * (2 * nf + 1);
bcos.resize(size_b);
bcos.setZero();
bsin.resize(size_b);
bsin.setZero();

const int size_a_temp = (mf + 1) * (2 * nf + 1) * (2 * nf + 1) * s_.nThetaEff;
actemp.resize(size_a_temp);
actemp.setZero();
astemp.resize(size_a_temp);
astemp.setZero();

bvec_sin.resize(mnpd);
bvec_sin.setZero();
amat_sin_sin.resize(mnpd * mnpd);
amat_sin_sin.setZero();

if (s_.lasym) {
gstore_asym.resize(s_.nThetaReduced * s_.nZeta);
gstore_asym.setZero();

bcos_asym.resize(size_b);
bcos_asym.setZero();
bsin_asym.resize(size_b);
bsin_asym.setZero();

actemp_cos.resize(size_a_temp);
actemp_cos.setZero();
astemp_cos.resize(size_a_temp);
astemp_cos.setZero();

bvec_cos.resize(mnpd);
bvec_cos.setZero();
amat_sin_cos.resize(mnpd * mnpd);
amat_sin_cos.setZero();
amat_cos_sin.resize(mnpd * mnpd);
amat_cos_sin.setZero();
amat_cos_cos.resize(mnpd * mnpd);
amat_cos_cos.setZero();
}

cosnv_scaled.resize(nf + 1, s_.nZeta);
sinnv_scaled.resize(nf + 1, s_.nZeta);
for (int n = 0; n < nf + 1; ++n) {
const double scale = 1.0 / fb_.nscale[n];
for (int k = 0; k < s_.nZeta; ++k) {
const int idx_nk = n * s_.nZeta + k;
cosnv_scaled(n, k) = fb_.cosnv[idx_nk] * scale;
sinnv_scaled(n, k) = fb_.sinnv[idx_nk] * scale;
}
}

cosmui_scaled.resize(s_.nThetaReduced, mf + 1);
sinmui_scaled.resize(s_.nThetaReduced, mf + 1);
for (int l = 0; l < s_.nThetaReduced; ++l) {
for (int m = 0; m < mf + 1; ++m) {
const int idx_lm = l * (s_.mnyq2 + 1) + m;
const double scale = 1.0 / fb_.mscale[m];
cosmui_scaled(l, m) = fb_.cosmui[idx_lm] * scale;
sinmui_scaled(l, m) = fb_.sinmui[idx_lm] * scale;
}
}
}

void LaplaceSolver::TransformGreensFunctionDerivative(
const Eigen::VectorXd& greenp) {
grpmn_sin.setZero();
if (s_.lasym) {
grpmn_cos.setZero();
}

Eigen::VectorXd g1_symm(nf + 1);
Eigen::VectorXd g2_symm(nf + 1);
Eigen::VectorXd kernel_odd(s_.nZeta);
Eigen::VectorXd g1_asym(nf + 1);
Eigen::VectorXd g2_asym(nf + 1);
Eigen::VectorXd kernel_even(s_.nZeta);

for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
const int klpRel = klp - tp_.ztMin;
const int klpOff = klpRel * s_.nThetaEven * s_.nZeta;

for (int l = 0; l < s_.nThetaReduced; ++l) {
const int lRev = (s_.nThetaEven - l) % s_.nThetaEven;

for (int k = 0; k < s_.nZeta; ++k) {
const int kRev = (s_.nZeta - k) % s_.nZeta;
const int kl = l * s_.nZeta + k;
const int klRev = lRev * s_.nZeta + kRev;

kernel_odd[k] = (greenp[klpOff + kl] - greenp[klpOff + klRev]) * 0.5;
if (s_.lasym) {
kernel_even[k] = (greenp[klpOff + kl] + greenp[klpOff + klRev]) * 0.5;
}
}

g1_symm.noalias() = cosnv_scaled * kernel_odd;
g2_symm.noalias() = sinnv_scaled * kernel_odd;
if (s_.lasym) {
g1_asym.noalias() = cosnv_scaled * kernel_even;
g2_asym.noalias() = sinnv_scaled * kernel_even;
}

for (int m = 0; m < mf + 1; ++m) {
const double cosmui = cosmui_scaled(l, m);
const double sinmui = sinmui_scaled(l, m);

for (int n = 0; n < nf + 1; ++n) {
const int idx_m_posn = (nf + n) * (mf + 1) + m;
const int idx_m_negn = (nf - n) * (mf + 1) + m;

const double gcos_symm = g1_symm[n] * sinmui;
const double gsin_symm = g2_symm[n] * cosmui;

grpmn_sin[idx_m_posn * numLocal + klpRel] += gcos_symm - gsin_symm;
if (n > 0) {
grpmn_sin[idx_m_negn * numLocal + klpRel] += gcos_symm + gsin_symm;
}

if (s_.lasym) {

const double gcos_asym = g1_asym[n] * cosmui;
const double gsin_asym = g2_asym[n] * sinmui;

grpmn_cos[idx_m_posn * numLocal + klpRel] += gcos_asym + gsin_asym;
if (n > 0) {
grpmn_cos[idx_m_negn * numLocal + klpRel] +=
gcos_asym - gsin_asym;
}
}
}
}
}
}
}

void LaplaceSolver::SymmetriseSourceTerm(const Eigen::VectorXd& gstore) {

for (int l = 0; l < s_.nThetaReduced; ++l) {
const int lRev = (s_.nThetaEven - l) % s_.nThetaEven;
for (int k = 0; k < s_.nZeta; ++k) {
const int kRev = (s_.nZeta - k) % s_.nZeta;

const int kl = l * s_.nZeta + k;
const int klRev = lRev * s_.nZeta + kRev;

gstore_symm[kl] = (gstore[kl] - gstore[klRev]) * 0.5;
if (s_.lasym) {
gstore_asym[kl] = (gstore[kl] + gstore[klRev]) * 0.5;
}
}
}
}

void LaplaceSolver::AccumulateFullGrpmn(
const Eigen::VectorXd& grpmn_sin_singular,
const Eigen::VectorXd& grpmn_cos_singular) {
const int mnpd = (mf + 1) * (2 * nf + 1);
const double inv_nfp = 1.0 / s_.nfp;

Eigen::Map<const Eigen::VectorXd> sin_singular(grpmn_sin_singular.data(),
mnpd * numLocal);
grpmn_sin += sin_singular * inv_nfp;

if (s_.lasym) {
Eigen::Map<const Eigen::VectorXd> cos_singular(grpmn_cos_singular.data(),
mnpd * numLocal);
grpmn_cos += cos_singular * inv_nfp;
}
}

void LaplaceSolver::PerformToroidalFourierTransforms() {
bcos.setZero();
bsin.setZero();
if (s_.lasym) {
bcos_asym.setZero();
bsin_asym.setZero();
}

Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
Eigen::RowMajor>>
gstore_mat(gstore_symm.data(), s_.nThetaReduced, s_.nZeta);

Eigen::MatrixXd bcos_mat = cosnv_scaled * gstore_mat.transpose();
Eigen::MatrixXd bsin_mat = sinnv_scaled * gstore_mat.transpose();

Eigen::MatrixXd bcos_mat_asym;
Eigen::MatrixXd bsin_mat_asym;
if (s_.lasym) {
Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
Eigen::RowMajor>>
gstore_asym_mat(gstore_asym.data(), s_.nThetaReduced, s_.nZeta);
bcos_mat_asym = cosnv_scaled * gstore_asym_mat.transpose();
bsin_mat_asym = sinnv_scaled * gstore_asym_mat.transpose();
}

for (int n = 0; n < nf + 1; ++n) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_l_posn = (nf + n) * s_.nThetaReduced + l;
bcos[idx_l_posn] = bcos_mat(n, l);
bsin[idx_l_posn] = bsin_mat(n, l);

if (n > 0) {
const int idx_l_negn = (nf - n) * s_.nThetaReduced + l;
bcos[idx_l_negn] = bcos_mat(n, l);
bsin[idx_l_negn] = -bsin_mat(n, l);
}

if (s_.lasym) {
bcos_asym[idx_l_posn] = bcos_mat_asym(n, l);
bsin_asym[idx_l_posn] = bsin_mat_asym(n, l);
if (n > 0) {
const int idx_l_negn = (nf - n) * s_.nThetaReduced + l;
bcos_asym[idx_l_negn] = bcos_mat_asym(n, l);
bsin_asym[idx_l_negn] = -bsin_mat_asym(n, l);
}
}
}
}

const int mnpd = (mf + 1) * (2 * nf + 1);
actemp.setZero();
astemp.setZero();
if (s_.lasym) {
actemp_cos.setZero();
astemp_cos.setZero();
}

for (int l = 0; l < s_.nThetaEff; ++l) {

for (int k = 0; k < s_.nZeta; ++k) {
const int klp = l * s_.nZeta + k;
if (klp < tp_.ztMin || klp >= tp_.ztMax) continue;

const int klpRel = klp - tp_.ztMin;

Eigen::VectorXd cosn_k = cosnv_scaled.col(k);
Eigen::VectorXd sinn_k = sinnv_scaled.col(k);

for (int mn = 0; mn < mnpd; ++mn) {
const double grpmn_sin_val = grpmn_sin[mn * numLocal + klpRel];
const double grpmn_cos_val =
s_.lasym ? grpmn_cos[mn * numLocal + klpRel] : 0.0;

for (int n = 0; n < nf + 1; ++n) {
const int idx_a_posn =
(mn * (2 * nf + 1) + (nf + n)) * s_.nThetaEff + l;
actemp[idx_a_posn] += cosn_k[n] * grpmn_sin_val;
astemp[idx_a_posn] += sinn_k[n] * grpmn_sin_val;
if (s_.lasym) {
actemp_cos[idx_a_posn] += cosn_k[n] * grpmn_cos_val;
astemp_cos[idx_a_posn] += sinn_k[n] * grpmn_cos_val;
}
}
}
}
}

for (int mn = 0; mn < mnpd; ++mn) {
for (int n = 1; n < nf + 1; ++n) {
for (int l = 0; l < s_.nThetaEff; ++l) {
const int idx_a_posn =
(mn * (2 * nf + 1) + (nf + n)) * s_.nThetaEff + l;
const int idx_a_negn =
(mn * (2 * nf + 1) + (nf - n)) * s_.nThetaEff + l;

actemp[idx_a_negn] = actemp[idx_a_posn];
astemp[idx_a_negn] = -astemp[idx_a_posn];
if (s_.lasym) {
actemp_cos[idx_a_negn] = actemp_cos[idx_a_posn];
astemp_cos[idx_a_negn] = -astemp_cos[idx_a_posn];
}
}
}
}
}

void LaplaceSolver::PerformPoloidalFourierTransforms() {
const int mnpd = (mf + 1) * (2 * nf + 1);
bvec_sin.setZero();
amat_sin_sin.setZero();
if (s_.lasym) {
bvec_cos.setZero();
amat_sin_cos.setZero();
amat_cos_sin.setZero();
amat_cos_cos.setZero();
}

for (int all_n = 0; all_n < 2 * nf + 1; ++all_n) {
Eigen::Map<const Eigen::VectorXd> bcos_n(
bcos.data() + all_n * s_.nThetaReduced, s_.nThetaReduced);
Eigen::Map<const Eigen::VectorXd> bsin_n(
bsin.data() + all_n * s_.nThetaReduced, s_.nThetaReduced);

Eigen::VectorXd result_sin =
sinmui_scaled.transpose() * bcos_n - cosmui_scaled.transpose() * bsin_n;

for (int m = 0; m < mf + 1; ++m) {
bvec_sin[all_n * (mf + 1) + m] = result_sin[m];
}

if (s_.lasym) {
Eigen::Map<const Eigen::VectorXd> bcos_asym_n(
bcos_asym.data() + all_n * s_.nThetaReduced, s_.nThetaReduced);
Eigen::Map<const Eigen::VectorXd> bsin_asym_n(
bsin_asym.data() + all_n * s_.nThetaReduced, s_.nThetaReduced);

Eigen::VectorXd result_cos = cosmui_scaled.transpose() * bcos_asym_n +
sinmui_scaled.transpose() * bsin_asym_n;
for (int m = 0; m < mf + 1; ++m) {
bvec_cos[all_n * (mf + 1) + m] = result_cos[m];
}
}
}

for (int mn = 0; mn < mnpd; ++mn) {
for (int all_n = 0; all_n < 2 * nf + 1; ++all_n) {
Eigen::VectorXd actemp_l(s_.nThetaReduced);
Eigen::VectorXd astemp_l(s_.nThetaReduced);

const int base_idx = (mn * (2 * nf + 1) + all_n) * s_.nThetaEff;
for (int l = 0; l < s_.nThetaReduced; ++l) {
actemp_l[l] = actemp[base_idx + l];
astemp_l[l] = astemp[base_idx + l];
}

if (!s_.lasym) {
Eigen::VectorXd result_ss = sinmui_scaled.transpose() * actemp_l -
cosmui_scaled.transpose() * astemp_l;
for (int m = 0; m < mf + 1; ++m) {
const int idx_amat = (all_n * (mf + 1) + m) * mnpd + mn;
amat_sin_sin[idx_amat] = result_ss[m];
}
} else {

Eigen::VectorXd ac_odd(s_.nThetaReduced), as_even(s_.nThetaReduced);
Eigen::VectorXd ac_even(s_.nThetaReduced), as_odd(s_.nThetaReduced);
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int rl = (s_.nThetaEven - l) % s_.nThetaEven;
const double a = actemp[base_idx + l];
const double ar = actemp[base_idx + rl];
const double b = astemp[base_idx + l];
const double br = astemp[base_idx + rl];
ac_odd[l] = 0.5 * (a - ar);
as_even[l] = 0.5 * (b + br);
ac_even[l] = 0.5 * (a + ar);
as_odd[l] = 0.5 * (b - br);
}

Eigen::VectorXd result_ss = sinmui_scaled.transpose() * ac_odd -
cosmui_scaled.transpose() * as_even;
Eigen::VectorXd result_sc = cosmui_scaled.transpose() * ac_even +
sinmui_scaled.transpose() * as_odd;
for (int m = 0; m < mf + 1; ++m) {
const int idx_amat = (all_n * (mf + 1) + m) * mnpd + mn;
amat_sin_sin[idx_amat] = result_ss[m];
amat_sin_cos[idx_amat] = result_sc[m];
}

Eigen::VectorXd cac_so(s_.nThetaReduced), cas_so(s_.nThetaReduced);
Eigen::VectorXd cac_co(s_.nThetaReduced), cas_co(s_.nThetaReduced);
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int rl = (s_.nThetaEven - l) % s_.nThetaEven;
const double ca = actemp_cos[base_idx + l];
const double car = actemp_cos[base_idx + rl];
const double cb = astemp_cos[base_idx + l];
const double cbr = astemp_cos[base_idx + rl];
cac_so[l] = 0.5 * (ca - car);
cas_so[l] = 0.5 * (cb + cbr);
cac_co[l] = 0.5 * (ca + car);
cas_co[l] = 0.5 * (cb - cbr);
}

Eigen::VectorXd result_cs = sinmui_scaled.transpose() * cac_so -
cosmui_scaled.transpose() * cas_so;
Eigen::VectorXd result_cc = cosmui_scaled.transpose() * cac_co +
sinmui_scaled.transpose() * cas_co;
for (int m = 0; m < mf + 1; ++m) {
const int idx_amat = (all_n * (mf + 1) + m) * mnpd + mn;
amat_cos_sin[idx_amat] = result_cs[m];
amat_cos_cos[idx_amat] = result_cc[m];
}
}
}
}
}

void LaplaceSolver::BuildMatrix() {
const int mnpd = (mf + 1) * (2 * nf + 1);
const int mnpd_dim = s_.lasym ? 2 * mnpd : mnpd;

const int matrix_size = mnpd_dim * mnpd_dim;
double* const slot = reduce_slots_.data() + tp_.get_thread_id() * matrix_size;
std::fill_n(slot, matrix_size, 0.0);
if (!s_.lasym) {
Eigen::Map<Eigen::VectorXd> slot_map(slot, mnpd * mnpd);
slot_map = amat_sin_sin;
} else {
const int stride = mnpd_dim;
for (int j = 0; j < mnpd; ++j) {
for (int i = 0; i < mnpd; ++i) {
const int src = j * mnpd + i;

slot[i + j * stride] = amat_sin_sin[src];

slot[i + (j + mnpd) * stride] = amat_sin_cos[src];

slot[(i + mnpd) + j * stride] = amat_cos_sin[src];

slot[(i + mnpd) + (j + mnpd) * stride] = amat_cos_cos[src];
}
}
}
SumOverThreads(slot, matrix_size, tp_.get_thread_id(), tp_.get_num_threads(),
reduce_slots_.data(), matrixShare.data());

#ifdef _OPENMP
#pragma omp single
#endif
{

for (int mnp = 0; mnp < mnpd; ++mnp) {
for (int all_n = 0; all_n < nf; ++all_n) {
const int m = 0;

if (!s_.lasym) {
matrixShare[(mnp * (2 * nf + 1) + all_n) * (mf + 1) + m] = 0.0;
} else {
const int row = all_n * (mf + 1) + m;
matrixShare[row + mnp * mnpd_dim] = 0.0;
matrixShare[row + (mnp + mnpd) * mnpd_dim] = 0.0;
matrixShare[(row + mnpd) + mnp * mnpd_dim] = 0.0;
matrixShare[(row + mnpd) + (mnp + mnpd) * mnpd_dim] = 0.0;
}
}
}

if (!s_.lasym) {
for (int mn = 0; mn < mnpd; ++mn) {
matrixShare[mn * mnpd + mn] += 0.5;
}
} else {

const int stride = mnpd_dim;
const int mn0 = nf * (mf + 1);

for (int mn = 0; mn < mnpd; ++mn) {
matrixShare[mn + mn * stride] += 0.5;
matrixShare[(mn + mnpd) + (mn + mnpd) * stride] += 0.5;
}
matrixShare[(mn0 + mnpd) + (mn0 + mnpd) * stride] += 0.5;
}
}
#ifdef _OPENMP
#pragma omp barrier
#endif
}

void LaplaceSolver::DecomposeMatrix() {
const int mnpd = (mf + 1) * (2 * nf + 1);
const int mnpd_dim = s_.lasym ? 2 * mnpd : mnpd;

Eigen::Map<const Eigen::MatrixXd> matrix_map(matrixShare.data(), mnpd_dim,
mnpd_dim);
lu_decomposition_->compute(matrix_map);

CHECK((lu_decomposition_->matrixLU().diagonal().array() != 0.0).all())
<< "singular matrix in LaplaceSolver::DecomposeMatrix";
}

void LaplaceSolver::SolveForPotential(
const Eigen::VectorXd& bvec_sin_singular,
const Eigen::VectorXd& bvec_cos_singular) {
const int mnpd = (mf + 1) * (2 * nf + 1);
const int mnpd_dim = s_.lasym ? 2 * mnpd : mnpd;
const double inv_nfp = 1.0 / s_.nfp;

double* const slot = reduce_slots_.data() + tp_.get_thread_id() * mnpd_dim;
std::fill_n(slot, mnpd_dim, 0.0);
{
Eigen::Map<Eigen::VectorXd> slot_sin(slot, mnpd);
Eigen::Map<const Eigen::VectorXd> singular_sin(bvec_sin_singular.data(),
mnpd);
slot_sin = bvec_sin + singular_sin * inv_nfp;

if (s_.lasym) {
Eigen::Map<Eigen::VectorXd> slot_cos(slot + mnpd, mnpd);
Eigen::Map<const Eigen::VectorXd> singular_cos(bvec_cos_singular.data(),
mnpd);
slot_cos = bvec_cos + singular_cos * inv_nfp;
}
}
SumOverThreads(slot, mnpd_dim, tp_.get_thread_id(), tp_.get_num_threads(),
reduce_slots_.data(), bvecShare.data());

#ifdef _OPENMP
#pragma omp single
#endif
{

for (int all_n = 0; all_n < nf; ++all_n) {
const int m = 0;
bvecShare[all_n * (mf + 1) + m] = 0.0;
if (s_.lasym) {
bvecShare[mnpd + all_n * (mf + 1) + m] = 0.0;
}
}

Eigen::Map<const Eigen::VectorXd> rhs(bvecShare.data(), mnpd_dim);
const Eigen::VectorXd solution = lu_decomposition_->solve(rhs);
Eigen::Map<Eigen::VectorXd>(bvecShare.data(), mnpd_dim) = solution;
}
#ifdef _OPENMP
#pragma omp barrier
#endif
}

}

// source: vmecpp/free_boundary/mgrid_provider/mgrid_provider.cc

#include <netcdf.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdio>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"

namespace vmecpp {

using netcdf_io::NetcdfReadArray3D;
using netcdf_io::NetcdfReadDouble;
using netcdf_io::NetcdfReadInt;
using netcdf_io::NetcdfReadString;

namespace {

absl::Status ValidateFieldContributionShape(
const std::vector<std::vector<std::vector<double> > >& field_contribution,
const std::string& variable_name, int num_phi, int num_z, int num_r) {
if (field_contribution.size() != static_cast<size_t>(num_phi)) {
return absl::InvalidArgumentError(
absl::StrFormat("Variable '%s' has %d phi slices, expected %d.",
variable_name, field_contribution.size(), num_phi));
}

for (int index_phi = 0; index_phi < num_phi; ++index_phi) {
if (field_contribution[index_phi].size() != static_cast<size_t>(num_z)) {
return absl::InvalidArgumentError(absl::StrFormat(
"Variable '%s' has %d z slices at phi index %d, expected %d.",
variable_name, field_contribution[index_phi].size(), index_phi,
num_z));
}
for (int index_z = 0; index_z < num_z; ++index_z) {
if (field_contribution[index_phi][index_z].size() !=
static_cast<size_t>(num_r)) {
return absl::InvalidArgumentError(absl::StrFormat(
"Variable '%s' has %d r points at phi index %d and z index %d, "
"expected %d.",
variable_name, field_contribution[index_phi][index_z].size(),
index_phi, index_z, num_r));
}
}
}
return absl::OkStatus();
}

}

void MGridProvider::ResetAccumulatedField() {
const int num_grid_points = numPhi * numZ * numR;
bR.setZero(num_grid_points);
bP.setZero(num_grid_points);
bZ.setZero(num_grid_points);
}

MGridProvider::MGridProvider() {
nfp = -1;

numR = -1;
minR = 0.0;
maxR = 0.0;
deltaR = 0.0;

numZ = -1;
minZ = 0.0;
maxZ = 0.0;
deltaZ = 0.0;

numPhi = -1;

nextcur = -1;

has_mgrid_loaded_ = false;

has_fixed_field_ = false;

mgrid_mode = "";
}

absl::Status MGridProvider::LoadFile(const std::filesystem::path& filename,
const Eigen::VectorXd& coil_currents) {
{
if (!std::filesystem::exists(filename)) {
return absl::NotFoundError(
absl::StrFormat("Could not find '%s'.", filename.string()));
}
std::ifstream fp(filename);
if (!fp.is_open()) {
return absl::PermissionDeniedError(absl::StrFormat(
"Mgrid file '%s' exists, but could not be opened for reading.",
filename.string()));
}
}

int ncid = 0;
if (nc_open(filename.c_str(), NC_NOWRITE, &ncid) != NC_NOERR) {
return absl::InternalError(
absl::StrFormat("NetCDF couldn't open '%s', despite passing "
"preconditions. The file may be corrupted.",
filename.string()));
}

auto with_context = [&filename](const absl::Status& s) {
return absl::Status(s.code(),
absl::StrFormat("While reading mgrid file '%s': %s",
filename.string(), s.message()));
};

absl::StatusOr<int> nfp_or = NetcdfReadInt(ncid, "nfp");

absl::StatusOr<int> num_r_or = NetcdfReadInt(ncid, "ir");
absl::StatusOr<double> min_r_or = NetcdfReadDouble(ncid, "rmin");
absl::StatusOr<double> max_r_or = NetcdfReadDouble(ncid, "rmax");

absl::StatusOr<int> num_z_or = NetcdfReadInt(ncid, "jz");
absl::StatusOr<double> min_z_or = NetcdfReadDouble(ncid, "zmin");
absl::StatusOr<double> max_z_or = NetcdfReadDouble(ncid, "zmax");

absl::StatusOr<int> num_phi_or = NetcdfReadInt(ncid, "kp");

absl::StatusOr<int> nextcur_or = NetcdfReadInt(ncid, "nextcur");

absl::StatusOr<std::string> mgrid_mode_or =
NetcdfReadString(ncid, "mgrid_mode");

absl::Status read_status;
read_status.Update(nfp_or.status());
read_status.Update(num_r_or.status());
read_status.Update(min_r_or.status());
read_status.Update(max_r_or.status());
read_status.Update(num_z_or.status());
read_status.Update(min_z_or.status());
read_status.Update(max_z_or.status());
read_status.Update(num_phi_or.status());
read_status.Update(nextcur_or.status());
read_status.Update(mgrid_mode_or.status());
if (!read_status.ok()) {
nc_close(ncid);
return with_context(read_status);
}

nfp = *nfp_or;

numR = *num_r_or;
minR = *min_r_or;
maxR = *max_r_or;
deltaR = (maxR - minR) / (numR - 1.0);

numZ = *num_z_or;
minZ = *min_z_or;
maxZ = *max_z_or;
deltaZ = (maxZ - minZ) / (numZ - 1.0);

numPhi = *num_phi_or;

nextcur = *nextcur_or;

coil_group_names.clear();
{
int id_coil_group = 0;
int num_dimensions = 0;
std::array<int, 2> coil_group_dimensions = {0, 0};
size_t num_groups = 0;
size_t string_width = 0;
if (nc_inq_varid(ncid, "coil_group", &id_coil_group) == NC_NOERR &&
nc_inq_varndims(ncid, id_coil_group, &num_dimensions) == NC_NOERR &&
num_dimensions == 2 &&
nc_inq_vardimid(ncid, id_coil_group, coil_group_dimensions.data()) ==
NC_NOERR &&
nc_inq_dimlen(ncid, coil_group_dimensions[0], &num_groups) ==
NC_NOERR &&
nc_inq_dimlen(ncid, coil_group_dimensions[1], &string_width) ==
NC_NOERR &&
string_width > 0) {

std::vector<char> raw(std::max(num_groups, static_cast<size_t>(nextcur)) *
string_width);
if (nc_get_var_text(ncid, id_coil_group, raw.data()) == NC_NOERR) {
coil_group_names.reserve(nextcur);
for (int i = 0; i < nextcur; ++i) {
std::string name(raw.data() + static_cast<size_t>(i) * string_width,
string_width);
size_t end = name.size();
while (end > 0 && name[end - 1] <= 0x20) {
--end;
}
name.erase(end);
coil_group_names.push_back(name);
}
}
}
}
if (coil_currents.size() != nextcur) {
nc_close(ncid);
return absl::InvalidArgumentError(
absl::StrFormat("Number of currents %d does not match number of mgrid "
"coil fields nextcur=%d.",
coil_currents.size(), nextcur));
}

mgrid_mode = *mgrid_mode_or;

ResetAccumulatedField();

for (int i = 0; i < nextcur; ++i) {

std::string br_variable = absl::StrFormat("br_%03d", i + 1);
absl::StatusOr<std::vector<std::vector<std::vector<double> > > >
b_r_contribution_or = NetcdfReadArray3D(ncid, br_variable);

std::string bp_variable = absl::StrFormat("bp_%03d", i + 1);
absl::StatusOr<std::vector<std::vector<std::vector<double> > > >
b_p_contribution_or = NetcdfReadArray3D(ncid, bp_variable);

std::string bz_variable = absl::StrFormat("bz_%03d", i + 1);
absl::StatusOr<std::vector<std::vector<std::vector<double> > > >
b_z_contribution_or = NetcdfReadArray3D(ncid, bz_variable);

absl::Status contribution_status;
contribution_status.Update(b_r_contribution_or.status());
contribution_status.Update(b_p_contribution_or.status());
contribution_status.Update(b_z_contribution_or.status());
if (!contribution_status.ok()) {
nc_close(ncid);
return with_context(contribution_status);
}

std::vector<std::vector<std::vector<double> > > b_r_contribution =
std::move(*b_r_contribution_or);
std::vector<std::vector<std::vector<double> > > b_p_contribution =
std::move(*b_p_contribution_or);
std::vector<std::vector<std::vector<double> > > b_z_contribution =
std::move(*b_z_contribution_or);

absl::Status shape_status;
shape_status.Update(ValidateFieldContributionShape(
b_r_contribution, br_variable, numPhi, numZ, numR));
shape_status.Update(ValidateFieldContributionShape(
b_p_contribution, bp_variable, numPhi, numZ, numR));
shape_status.Update(ValidateFieldContributionShape(
b_z_contribution, bz_variable, numPhi, numZ, numR));
if (!shape_status.ok()) {
nc_close(ncid);
return with_context(shape_status);
}

AccumulateCircuit(coil_currents[i], [&](int linear_index) {
const int index_r = linear_index % numR;
const int index_z = (linear_index / numR) % numZ;
const int index_phi = linear_index / (numZ * numR);
return std::make_tuple(b_r_contribution[index_phi][index_z][index_r],
b_p_contribution[index_phi][index_z][index_r],
b_z_contribution[index_phi][index_z][index_r]);
});
}

if (nc_close(ncid) != NC_NOERR) {
return absl::InternalError("Failed to close NetCDF file.");
}

has_mgrid_loaded_ = true;
has_fixed_field_ = false;

return absl::Status();
}

absl::Status MGridProvider::LoadFields(
const makegrid::MagneticFieldResponseTable& magnetic_response_table,
const Eigen::VectorXd& coil_currents) {
const auto& mgrid_params = magnetic_response_table.parameters;
if (coil_currents.size() != magnetic_response_table.b_p.rows()) {
return absl::InvalidArgumentError(absl::StrFormat(
"Number of currents %d does not match number of coil fields in the "
"response table %d.",
coil_currents.size(), magnetic_response_table.b_p.rows()));
}

nfp = mgrid_params.number_of_field_periods;

numR = mgrid_params.number_of_r_grid_points;
minR = mgrid_params.r_grid_minimum;
maxR = mgrid_params.r_grid_maximum;
deltaR = (maxR - minR) / (numR - 1.0);

numZ = mgrid_params.number_of_z_grid_points;
minZ = mgrid_params.z_grid_minimum;
maxZ = mgrid_params.z_grid_maximum;
deltaZ = (maxZ - minZ) / (numZ - 1.0);

numPhi = mgrid_params.number_of_phi_grid_points;

nextcur = static_cast<int>(coil_currents.size());

coil_group_names.clear();

if (mgrid_params.normalize_by_currents) {
mgrid_mode = "S";
} else {
mgrid_mode = "R";
}

ResetAccumulatedField();

for (int i = 0; i < nextcur; ++i) {
AccumulateCircuit(coil_currents[i], [&](int linear_index) {
return std::make_tuple(magnetic_response_table.b_r(i, linear_index),
magnetic_response_table.b_p(i, linear_index),
magnetic_response_table.b_z(i, linear_index));
});
}

has_mgrid_loaded_ = true;
has_fixed_field_ = false;

return absl::OkStatus();
}

void MGridProvider::SetFixedMagneticField(const Eigen::VectorXd& fixed_br,
const Eigen::VectorXd& fixed_bp,
const Eigen::VectorXd& fixed_bz) {

fixed_br_ = fixed_br;
fixed_bp_ = fixed_bp;
fixed_bz_ = fixed_bz;

has_mgrid_loaded_ = true;
has_fixed_field_ = true;
}

absl::Status MGridProvider::interpolate(int ztMin, int ztMax, int nZeta,
int nZnT, const Eigen::VectorXd& rLCFS,
const Eigen::VectorXd& zLCFS,
Eigen::VectorXd& m_interpBr,
Eigen::VectorXd& m_interpBp,
Eigen::VectorXd& m_interpBz) const {
CHECK(has_mgrid_loaded_) << "no mgrid loaded";

if (has_fixed_field_) {

for (int kl = ztMin; kl < ztMax; ++kl) {
m_interpBr[kl - ztMin] = fixed_br_[kl];
m_interpBp[kl - ztMin] = fixed_bp_[kl];
m_interpBz[kl - ztMin] = fixed_bz_[kl];
}

return absl::OkStatus();
}

bool exceedGridSizeR = false;
bool exceedGridSizeZ = false;
for (int kl = ztMin; kl < ztMax; ++kl) {
int k = kl % nZeta;

if (rLCFS[kl] < minR || rLCFS[kl] > maxR) {
exceedGridSizeR = true;
}
if (zLCFS[kl] < minZ || zLCFS[kl] > maxZ) {
exceedGridSizeZ = true;
}

double r = std::max(minR, std::min(rLCFS[kl], maxR));
double z = std::max(minZ, std::min(zLCFS[kl], maxZ));

const int r_nodes = std::min(4, numR);
const int z_nodes = std::min(4, numZ);
const double r_index = (r - minR) / deltaR;
const double z_index = (z - minZ) / deltaZ;
const int r_start =
std::clamp(static_cast<int>(floor(r_index)) - 1, 0, numR - r_nodes);
const int z_start =
std::clamp(static_cast<int>(floor(z_index)) - 1, 0, numZ - z_nodes);
const auto weights = [](double u, int nodes) -> std::array<double, 4> {
if (nodes == 4) {
return {-(u - 1) * (u - 2) * (u - 3) / 6, u * (u - 2) * (u - 3) / 2,
-u * (u - 1) * (u - 3) / 2, u * (u - 1) * (u - 2) / 6};
}
std::array<double, 4> result{};
for (int i = 0; i < nodes; ++i) {
result[i] = 1.0;
for (int j = 0; j < nodes; ++j) {
if (i != j) result[i] *= (u - j) / (i - j);
}
}
return result;
};
const auto r_weights = weights(r_index - r_start, r_nodes);
const auto z_weights = weights(z_index - z_start, z_nodes);
double br = 0.0;
double bp = 0.0;
double bz = 0.0;
for (int j = 0; j < z_nodes; ++j) {
for (int i = 0; i < r_nodes; ++i) {
const int index = (k * numZ + z_start + j) * numR + r_start + i;
const double weight = r_weights[i] * z_weights[j];
br += weight * bR[index];
bp += weight * bP[index];
bz += weight * bZ[index];
}
}
m_interpBr[kl - ztMin] = br;
m_interpBp[kl - ztMin] = bp;
m_interpBz[kl - ztMin] = bz;
}

absl::Status status = absl::OkStatus();
if (exceedGridSizeR || exceedGridSizeZ) {

double min_r = DBL_MAX;
double max_r = -DBL_MAX;
double min_z = DBL_MAX;
double max_z = -DBL_MAX;
for (int kl = 0; kl < nZnT; ++kl) {
min_r = std::min(min_r, rLCFS[kl]);
max_r = std::max(max_r, rLCFS[kl]);
min_z = std::min(min_z, zLCFS[kl]);
max_z = std::max(max_z, zLCFS[kl]);
}

std::string exceeded_extents;
if (exceedGridSizeR) {
exceeded_extents += absl::StrFormat(
" R: boundary [% .6e, % .6e] against grid [% .6e, % .6e].", min_r,
max_r, minR, maxR);
}
if (exceedGridSizeZ) {
exceeded_extents += absl::StrFormat(
" Z: boundary [% .6e, % .6e] against grid [% .6e, % .6e].", min_z,
max_z, minZ, maxZ);
}

status = absl::FailedPreconditionError(absl::StrFormat(
"MGridProvider::interpolate: the plasma boundary exceeded the vacuum "
"field grid, so the interpolated field was clamped to the grid edge "
"and is not physically meaningful.%s Enlarge the mgrid domain so that "
"it contains the plasma boundary at every iteration.",
exceeded_extents));
}

#ifdef _OPENMP
#pragma omp barrier
#endif

return status;
}

}

// source: vmecpp/free_boundary/nestor/nestor.cc

// header: vmecpp/free_boundary/nestor/nestor.h

#ifndef VMECPP_FREE_BOUNDARY_NESTOR_NESTOR_H_
#define VMECPP_FREE_BOUNDARY_NESTOR_NESTOR_H_

#include <Eigen/Dense>
#include <span>

// header: vmecpp/free_boundary/free_boundary_base/free_boundary_base.h

#ifndef VMECPP_FREE_BOUNDARY_FREE_BOUNDARY_BASE_FREE_BOUNDARY_BASE_H_
#define VMECPP_FREE_BOUNDARY_FREE_BOUNDARY_BASE_FREE_BOUNDARY_BASE_H_

#include <span>
#include <vector>

#include "absl/status/statusor.h"

namespace vmecpp {

class FreeBoundaryBase {
public:
virtual ~FreeBoundaryBase() = default;

FreeBoundaryBase(const Sizes* s, const TangentialPartitioning* tp,
const MGridProvider* mgrid, std::span<double> bSqVacShare,
std::span<double> vacuum_b_r_share,
std::span<double> vacuum_b_phi_share,
std::span<double> vacuum_b_z_share)
: s_(*s),
fb_(&s_),
tp_(*tp),
sg_(s, &fb_, tp),
ef_(s, tp, &sg_, mgrid),
bSqVacShare(bSqVacShare),
vacuum_b_r_share_(vacuum_b_r_share),
vacuum_b_phi_share_(vacuum_b_phi_share),
vacuum_b_z_share_(vacuum_b_z_share) {}

virtual absl::StatusOr<bool> update(
const std::span<const double> rCC, const std::span<const double> rSS,
const std::span<const double> rSC, const std::span<const double> rCS,
const std::span<const double> zSC, const std::span<const double> zCS,
const std::span<const double> zCC, const std::span<const double> zSS,
int signOfJacobian, const std::span<const double> rAxis,
const std::span<const double> zAxis, double* bSubUVac, double* bSubVVac,
double netToroidalCurrent, int m_ivacskip,
const VmecCheckpoint& vmec_checkpoint = VmecCheckpoint::NONE,
bool at_checkpoint_iteration = false) = 0;

const SurfaceGeometry& GetSurfaceGeometry() const { return sg_; }
const ExternalMagneticField& GetExternalMagneticField() const { return ef_; }

protected:
const Sizes& s_;
const FourierBasisFastToroidal fb_;
const TangentialPartitioning& tp_;

SurfaceGeometry sg_;
ExternalMagneticField ef_;

std::span<double> bSqVacShare;

std::span<double> vacuum_b_r_share_;

std::span<double> vacuum_b_phi_share_;

std::span<double> vacuum_b_z_share_;
};

}

#endif

// header: vmecpp/free_boundary/regularized_integrals/regularized_integrals.h

#ifndef VMECPP_FREE_BOUNDARY_REGULARIZED_INTEGRALS_REGULARIZED_INTEGRALS_H_
#define VMECPP_FREE_BOUNDARY_REGULARIZED_INTEGRALS_REGULARIZED_INTEGRALS_H_

#include <Eigen/Dense>

namespace vmecpp {

class RegularizedIntegrals {
public:
RegularizedIntegrals(const Sizes* s, const TangentialPartitioning* tp,
const SurfaceGeometry* sg);

void update(const Eigen::VectorXd& bDotN);

Eigen::VectorXd gsave;
Eigen::VectorXd dsave;

Eigen::VectorXd tanu;
Eigen::VectorXd tanv;

Eigen::VectorXd greenp;
Eigen::VectorXd gstore;

private:

static constexpr int kAxisymmetricToroidalImages = 64;

const Sizes& s_;
const TangentialPartitioning& tp_;
const SurfaceGeometry& sg_;

int nvper_;

std::vector<double> tanv_per_;

void computeConstants();

void updateAxisymmetric(const Eigen::VectorXd& bDotN);
};

}

#endif

// header: vmecpp/free_boundary/singular_integrals/singular_integrals.h

#ifndef VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_
#define VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_

#include <Eigen/Dense>
#include <vector>

namespace vmecpp {

inline constexpr int kChebyshevMomentBatch = 8;

struct ChebyshevMomentWorkspace {

std::vector<double> factor_sub2;
std::vector<double> factor_sub1;
std::vector<double> factor_diag;
std::vector<double> factor_sup1;
std::vector<double> factor_sup2;
std::vector<double> factor_rhs;

Eigen::Array<double, kChebyshevMomentBatch, Eigen::Dynamic> upper1;
Eigen::Array<double, kChebyshevMomentBatch, Eigen::Dynamic> upper2;
Eigen::Array<double, kChebyshevMomentBatch, Eigen::Dynamic> moments;
};

class SingularIntegrals {
public:
SingularIntegrals(const Sizes* s, const FourierBasisFastToroidal* fb,
const TangentialPartitioning* tp, const SurfaceGeometry* sg,
int nf, int mf);

void update(const Eigen::VectorXd& bDotN, bool fullUpdate);

int numSC;
int numCS;
int nzLen;

Eigen::VectorXd chebyshev_coefficients;

Eigen::VectorXd ap;
Eigen::VectorXd am;
Eigen::VectorXd d;
Eigen::VectorXd sqrtc2;
Eigen::VectorXd sqrta2;
Eigen::VectorXd delta4;

Eigen::VectorXd Ap;
Eigen::VectorXd Am;
Eigen::VectorXd D;

Eigen::VectorXd R1p;
Eigen::VectorXd R1m;
Eigen::VectorXd R0p;
Eigen::VectorXd R0m;
Eigen::VectorXd Ra1p;
Eigen::VectorXd Ra1m;

std::vector<Eigen::VectorXd> chebyshev_moments_p;
std::vector<Eigen::VectorXd> chebyshev_moments_m;

std::vector<Eigen::VectorXd> chebyshev_s_moments_p;
std::vector<Eigen::VectorXd> chebyshev_s_moments_m;

Eigen::VectorXd bvec_sin;

Eigen::VectorXd bvec_cos;

Eigen::VectorXd grpmn_sin;

Eigen::VectorXd grpmn_cos;

void prepareUpdate(const Eigen::VectorXd& a, const Eigen::VectorXd& b2,
const Eigen::VectorXd& c, const Eigen::VectorXd& A,
const Eigen::VectorXd& B2, const Eigen::VectorXd& C,
bool fullUpdate);

private:
const Sizes& s_;
const FourierBasisFastToroidal& fb_;
const TangentialPartitioning& tp_;
const SurfaceGeometry& sg_;

void computeCoefficients();

void performUpdate(const Eigen::VectorXd& bDotN, bool fullUpdate);

int nf;
int mf;

ChebyshevMomentWorkspace moment_workspace_;
};

}

#endif

namespace vmecpp {

class Nestor : public FreeBoundaryBase {
public:
Nestor(const Sizes* s, const TangentialPartitioning* tp,
const MGridProvider* mgrid, std::span<double> matrixShare,
std::span<double> bvecShare, std::span<double> bSqVacShare,
Eigen::PartialPivLU<Eigen::MatrixXd>* lu_decomposition,
std::span<double> vacuum_b_r_share,
std::span<double> vacuum_b_phi_share,
std::span<double> vacuum_b_z_share, std::span<double> reduce_slots);

absl::StatusOr<bool> update(
const std::span<const double> rCC, const std::span<const double> rSS,
const std::span<const double> rSC, const std::span<const double> rCS,
const std::span<const double> zSC, const std::span<const double> zCS,
const std::span<const double> zCC, const std::span<const double> zSS,
int signOfJacobian, const std::span<const double> rAxis,
const std::span<const double> zAxis, double* bSubUVac, double* bSubVVac,
double netToroidalCurrent, int ivacskip,
const VmecCheckpoint& vmec_checkpoint = VmecCheckpoint::NONE,
bool at_checkpoint_iteration = false) final;

const SingularIntegrals& GetSingularIntegrals() const;
const RegularizedIntegrals& GetRegularizedIntegrals() const;
const LaplaceSolver& GetLaplaceSolver() const;

Eigen::VectorXd potU;
Eigen::VectorXd potV;

Eigen::VectorXd bSubU;
Eigen::VectorXd bSubV;

private:

const int nf;

const int mf;

SingularIntegrals si_;
RegularizedIntegrals ri_;
LaplaceSolver ls_;

std::span<double> bvecShare;

std::span<double> reduce_slots_;
};

}

#endif

#include <cmath>

namespace vmecpp {

Nestor::Nestor(const Sizes* s, const TangentialPartitioning* tp,
const MGridProvider* mgrid, std::span<double> matrixShare,
std::span<double> bvecShare, std::span<double> bSqVacShare,
Eigen::PartialPivLU<Eigen::MatrixXd>* lu_decomposition,
std::span<double> vacuum_b_r_share,
std::span<double> vacuum_b_phi_share,
std::span<double> vacuum_b_z_share,
std::span<double> reduce_slots)
: FreeBoundaryBase(s, tp, mgrid, bSqVacShare, vacuum_b_r_share,
vacuum_b_phi_share, vacuum_b_z_share),
nf(s_.ntor),
mf(s_.mpol + 1),
si_(s, &fb_, tp, &sg_, nf, mf),
ri_(s, tp, &sg_),
ls_(s, &fb_, tp, nf, mf, matrixShare, lu_decomposition, bvecShare,
reduce_slots),
bvecShare(bvecShare),
reduce_slots_(reduce_slots) {
int numLocal = tp_.ztMax - tp_.ztMin;

potU.setZero(numLocal);
potV.setZero(numLocal);

bSubU.setZero(numLocal);
bSubV.setZero(numLocal);
}

absl::StatusOr<bool> Nestor::update(
const std::span<const double> rCC, const std::span<const double> rSS,
const std::span<const double> rSC, const std::span<const double> rCS,
const std::span<const double> zSC, const std::span<const double> zCS,
const std::span<const double> zCC, const std::span<const double> zSS,
int signOfJacobian, const std::span<const double> rAxis,
const std::span<const double> zAxis, double* bSubUVac, double* bSubVVac,
double netToroidalCurrent, int ivacskip,
const VmecCheckpoint& vmec_checkpoint, bool at_checkpoint_iteration) {
if (vmec_checkpoint == VmecCheckpoint::VAC1_VACUUM &&
at_checkpoint_iteration) {
return true;
}

bool fullUpdate = (ivacskip == 0);

sg_.update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, signOfJacobian,
fullUpdate);
if (vmec_checkpoint == VmecCheckpoint::VAC1_SURFACE &&
at_checkpoint_iteration) {
return true;
}

const absl::Status external_field_status =
ef_.update(rAxis, zAxis, netToroidalCurrent);
if (vmec_checkpoint == VmecCheckpoint::VAC1_BEXTERN &&
at_checkpoint_iteration) {
return true;
}

si_.update(ef_.bDotN, fullUpdate);
if (vmec_checkpoint == VmecCheckpoint::VAC1_ANALYT &&
at_checkpoint_iteration) {
return true;
}

if (fullUpdate) {
ri_.update(ef_.bDotN);
if (vmec_checkpoint == VmecCheckpoint::VAC1_GREENF &&
at_checkpoint_iteration) {
return true;
}

ls_.TransformGreensFunctionDerivative(ri_.greenp);
if (vmec_checkpoint == VmecCheckpoint::VAC1_FOURP &&
at_checkpoint_iteration) {
return true;
}

ls_.SymmetriseSourceTerm(ri_.gstore);
if (vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_SYMM &&
at_checkpoint_iteration) {
return true;
}

ls_.AccumulateFullGrpmn(si_.grpmn_sin, si_.grpmn_cos);
ls_.PerformToroidalFourierTransforms();
if (vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_KV_DFT &&
at_checkpoint_iteration) {
return true;
}

ls_.PerformPoloidalFourierTransforms();
ls_.BuildMatrix();
if (vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_KU_DFT &&
at_checkpoint_iteration) {
return true;
}

#ifdef _OPENMP
#pragma omp single
#endif
ls_.DecomposeMatrix();
#ifdef _OPENMP
#pragma omp barrier
#endif
}

if ((vmec_checkpoint == VmecCheckpoint::VAC1_GREENF ||
vmec_checkpoint == VmecCheckpoint::VAC1_FOURP ||
vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_SYMM ||
vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_KV_DFT ||
vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_KU_DFT ||
vmec_checkpoint == VmecCheckpoint::UPDATE_TCON) &&
at_checkpoint_iteration) {
return true;
}

ls_.SolveForPotential(si_.bvec_sin, si_.bvec_cos);

if (vmec_checkpoint == VmecCheckpoint::VAC1_SOLVER &&
at_checkpoint_iteration) {
return true;
}

const int mnpd = (mf + 1) * (2 * nf + 1);

potU.setZero();
potV.setZero();

for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
const int l = kl / s_.nZeta;
const int k = kl % s_.nZeta;

const int lr = (l < s_.nThetaReduced) ? l : (s_.nThetaEven - l);
const double sgnmu = (l < s_.nThetaReduced) ? 1.0 : -1.0;
for (int mn = 0; mn < mnpd; ++mn) {
const int n = mn / (mf + 1) - nf;
const int m = mn % (mf + 1);

const int abs_n = std::abs(n);
const int sign_n = signum(n);

const int idx_lm = lr * (s_.mnyq2 + 1) + m;
const double cosmu = fb_.cosmu[idx_lm] / fb_.mscale[m];
const double sinmu = sgnmu * fb_.sinmu[idx_lm] / fb_.mscale[m];

const int idx_nk = abs_n * s_.nZeta + k;
const double cosnv = fb_.cosnv[idx_nk] / fb_.nscale[abs_n];
const double sinnv = fb_.sinnv[idx_nk] / fb_.nscale[abs_n];

const double cos_mu_nv = cosmu * cosnv + sign_n * sinmu * sinnv;

potU[kl - tp_.ztMin] += bvecShare[mn] * m * cos_mu_nv;
potV[kl - tp_.ztMin] += bvecShare[mn] * (-n * s_.nfp) * cos_mu_nv;

if (s_.lasym) {
const double sin_mu_nv = sinmu * cosnv - sign_n * cosmu * sinnv;
const double potcos = bvecShare[mnpd + mn];
potU[kl - tp_.ztMin] -= potcos * m * sin_mu_nv;
potV[kl - tp_.ztMin] -= potcos * (-n * s_.nfp) * sin_mu_nv;
}
}
}

double local_bSubUVac = 0.0;
double local_bSubVVac = 0.0;
for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
bSubU[kl - tp_.ztMin] = potU[kl - tp_.ztMin] + ef_.bSubU[kl - tp_.ztMin];
bSubV[kl - tp_.ztMin] = potV[kl - tp_.ztMin] + ef_.bSubV[kl - tp_.ztMin];

int l = kl / s_.nZeta;
local_bSubUVac += bSubU[kl - tp_.ztMin] * s_.wInt[l];
local_bSubVVac += bSubV[kl - tp_.ztMin] * s_.wInt[l];
}
local_bSubUVac *= signOfJacobian * 2.0 * M_PI;

#ifdef _OPENMP
#pragma omp single
#endif
{
*bSubUVac = 0.0;
*bSubVVac = 0.0;
}

SumOverThreads(&local_bSubUVac, 1, tp_.get_thread_id(), tp_.get_num_threads(),
reduce_slots_.data(), bSubUVac);
SumOverThreads(&local_bSubVVac, 1, tp_.get_thread_id(), tp_.get_num_threads(),
reduce_slots_.data(), bSubVVac);

for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {

double guu = sg_.guu[kl - tp_.ztMin];
double guv = sg_.guv[kl - tp_.ztMin] * s_.nfp * 0.5;
double gvv = sg_.gvv[kl - tp_.ztMin] * s_.nfp * s_.nfp;

double det = guu * gvv - guv * guv;

double bSupU =
(gvv * bSubU[kl - tp_.ztMin] - guv * bSubV[kl - tp_.ztMin]) / det;
double bSupV =
(-guv * bSubU[kl - tp_.ztMin] + guu * bSubV[kl - tp_.ztMin]) / det;

bSqVacShare[kl] =
(bSubU[kl - tp_.ztMin] * bSupU + bSubV[kl - tp_.ztMin] * bSupV) * 0.5;

const int derivOffset = s_.lasym ? 0 : tp_.ztMin;
vacuum_b_r_share_[kl] =
sg_.rub[kl - derivOffset] * bSupU + sg_.rvb[kl - derivOffset] * bSupV;
vacuum_b_phi_share_[kl] = sg_.r1b[kl] * bSupV;
vacuum_b_z_share_[kl] =
sg_.zub[kl - derivOffset] * bSupU + sg_.zvb[kl - derivOffset] * bSupV;
}

if (vmec_checkpoint == VmecCheckpoint::VAC1_BSQVAC &&
at_checkpoint_iteration) {
return true;
}

#ifdef _OPENMP
#pragma omp barrier
#endif

if (!external_field_status.ok()) {
return external_field_status;
}

return false;
}

const SingularIntegrals& Nestor::GetSingularIntegrals() const {
return si_;
}

const RegularizedIntegrals& Nestor::GetRegularizedIntegrals() const {
return ri_;
}

const LaplaceSolver& Nestor::GetLaplaceSolver() const {
return ls_;
}

}

// source: vmecpp/free_boundary/only_coils/only_coils.cc

// header: vmecpp/free_boundary/only_coils/only_coils.h

#ifndef VMECPP_FREE_BOUNDARY_ONLY_COILS_ONLY_COILS_H_
#define VMECPP_FREE_BOUNDARY_ONLY_COILS_ONLY_COILS_H_

#include <span>

namespace vmecpp {

class OnlyCoils : public FreeBoundaryBase {
public:
OnlyCoils(const Sizes* s, const TangentialPartitioning* tp,
const MGridProvider* mgrid, std::span<double> bSqVacShare,
std::span<double> vacuum_b_r_share,
std::span<double> vacuum_b_phi_share,
std::span<double> vacuum_b_z_share, std::span<double> reduce_slots);

absl::StatusOr<bool> update(
const std::span<const double> rCC, const std::span<const double> rSS,
const std::span<const double> rSC, const std::span<const double> rCS,
const std::span<const double> zSC, const std::span<const double> zCS,
const std::span<const double> zCC, const std::span<const double> zSS,
int signOfJacobian, const std::span<const double> rAxis,
const std::span<const double> zAxis, double* bSubUVac, double* bSubVVac,
double netToroidalCurrent, int ivacskip,
const VmecCheckpoint& vmec_checkpoint = VmecCheckpoint::NONE,
bool at_checkpoint_iteration = false) final;

private:

std::span<double> reduce_slots_;
};

}

#endif

namespace vmecpp {

OnlyCoils::OnlyCoils(const Sizes* s, const TangentialPartitioning* tp,
const MGridProvider* mgrid, std::span<double> bSqVacShare,
std::span<double> vacuum_b_r_share,
std::span<double> vacuum_b_phi_share,
std::span<double> vacuum_b_z_share,
std::span<double> reduce_slots)
: FreeBoundaryBase(s, tp, mgrid, bSqVacShare, vacuum_b_r_share,
vacuum_b_phi_share, vacuum_b_z_share),
reduce_slots_(reduce_slots) {}

absl::StatusOr<bool> OnlyCoils::update(
const std::span<const double> rCC, const std::span<const double> rSS,
const std::span<const double> rSC, const std::span<const double> rCS,
const std::span<const double> zSC, const std::span<const double> zCS,
const std::span<const double> zCC, const std::span<const double> zSS,
int signOfJacobian, const std::span<const double> rAxis,
const std::span<const double> zAxis, double* bSubUVac, double* bSubVVac,
double netToroidalCurrent, int ivacskip,
const VmecCheckpoint& vmec_checkpoint, bool at_checkpoint_iteration) {

bool full_update = false;

sg_.update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, signOfJacobian,
full_update);

const absl::Status external_field_status = ef_.update(rAxis, zAxis, 0.0);

double local_bsubuvac = 0.0;
double local_bsubvvac = 0.0;
for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
int l = kl / s_.nZeta;
local_bsubuvac += ef_.bSubU[kl - tp_.ztMin] * s_.wInt[l];
local_bsubvvac += ef_.bSubV[kl - tp_.ztMin] * s_.wInt[l];
}
local_bsubuvac *= signOfJacobian * 2.0 * M_PI;

#ifdef _OPENMP
#pragma omp single
#endif
{
*bSubUVac = 0.0;
*bSubVVac = 0.0;
}

SumOverThreads(&local_bsubuvac, 1, tp_.get_thread_id(), tp_.get_num_threads(),
reduce_slots_.data(), bSubUVac);
SumOverThreads(&local_bsubvvac, 1, tp_.get_thread_id(), tp_.get_num_threads(),
reduce_slots_.data(), bSubVVac);

for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {

vacuum_b_r_share_[kl] = ef_.interpBr[kl - tp_.ztMin];
vacuum_b_phi_share_[kl] = ef_.interpBp[kl - tp_.ztMin];
vacuum_b_z_share_[kl] = ef_.interpBz[kl - tp_.ztMin];

bSqVacShare[kl] = 0.5 * (vacuum_b_r_share_[kl] * vacuum_b_r_share_[kl] +
vacuum_b_phi_share_[kl] * vacuum_b_phi_share_[kl] +
vacuum_b_z_share_[kl] * vacuum_b_z_share_[kl]);
}

#ifdef _OPENMP
#pragma omp barrier
#endif

if (!external_field_status.ok()) {
return external_field_status;
}

return false;
}

}

// source: vmecpp/free_boundary/regularized_integrals/regularized_integrals.cc

#include <algorithm>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"

namespace vmecpp {

RegularizedIntegrals::RegularizedIntegrals(const Sizes* s,
const TangentialPartitioning* tp,
const SurfaceGeometry* sg)
: s_(*s), tp_(*tp), sg_(*sg) {
gsave.resize(s_.nThetaEven * s_.nZeta);
dsave.resize(s_.nThetaEven * s_.nZeta);

tanu.resize(s_.nThetaEven);
tanv.resize(s_.nZeta);

int numLocal = tp_.ztMax - tp_.ztMin;
greenp.resize(numLocal * s_.nThetaEven * s_.nZeta);

gstore.resize(s_.nThetaEven * s_.nZeta);

computeConstants();
}

void RegularizedIntegrals::computeConstants() {
const double epsTan = 1.0e-15;
const double bigNo = 1.0e50;

for (int l = 0; l < s_.nThetaEven; ++l) {
const double argu = M_PI / s_.nThetaEven * l;
if (std::abs(argu - 0.5 * M_PI) < epsTan) {

tanu[l] = bigNo;
} else {
tanu[l] = 2.0 * std::tan(argu);
}
}

for (int k = 0; k < s_.nZeta; ++k) {
const double argv = M_PI / s_.nZeta * k;
if (std::abs(argv - 0.5 * M_PI) < epsTan) {

tanv[k] = bigNo;
} else {
tanv[k] = 2.0 * std::tan(argv);
}
}

nvper_ = (s_.nZeta == 1) ? kAxisymmetricToroidalImages : s_.nfp;
if (s_.nZeta == 1) {
tanv_per_.resize(nvper_);
for (int p = 0; p < nvper_; ++p) {
const double argv = M_PI * p / nvper_;
if (std::abs(argv - 0.5 * M_PI) < epsTan) {

tanv_per_[p] = bigNo;
} else {
tanv_per_[p] = 2.0 * std::tan(argv);
}
}
}
}

void RegularizedIntegrals::update(const Eigen::VectorXd& bDotN) {
if (s_.nZeta == 1) {

updateAxisymmetric(bDotN);
return;
}

const int theta_by_nzeta = s_.nThetaEven * s_.nZeta;
const double twopidivnfp = 2.0 * M_PI / s_.nfp;

greenp.setZero();
gstore.setZero();

std::vector<double> ga1_buf(s_.nZeta);
std::vector<double> ga2_buf(s_.nZeta);
std::vector<double> htemp_buf(s_.nZeta);
std::vector<double> ftemp_buf(s_.nZeta);

for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
const int ip_idx_base = (klp - tp_.ztMin) * theta_by_nzeta;
int lp = klp / s_.nZeta;
int kp = klp % s_.nZeta;

double bexni = bDotN[klp - tp_.ztMin] * s_.wInt[lp];

double xp = sg_.rcosuv[klp];
double yp = sg_.rsinuv[klp];

int p = 0;

double xper = xp * sg_.cos_per[p] - yp * sg_.sin_per[p];
double yper = xp * sg_.sin_per[p] + yp * sg_.cos_per[p];

double sxsave =
(sg_.snr[klp - tp_.ztMin] * xper - sg_.snv[klp - tp_.ztMin] * yper) /
sg_.r1b[klp];
double sysave =
(sg_.snr[klp - tp_.ztMin] * yper + sg_.snv[klp - tp_.ztMin] * xper) /
sg_.r1b[klp];

for (int kl = 0; kl < theta_by_nzeta; ++kl) {
gsave[kl] = sg_.rzb2[klp] + sg_.rzb2[kl] - 2 * sg_.z1b[kl] * sg_.z1b[klp];
}

for (int kl = 0; kl < theta_by_nzeta; ++kl) {
dsave[kl] =
sg_.drv[klp - tp_.ztMin] + sg_.z1b[kl] * sg_.snz[klp - tp_.ztMin];
}

auto do_loop = [&](int kl_start, int kl_end) {

for (int kl = kl_start; kl < kl_end;) {

const int l = kl / s_.nZeta;
const int delta_l = (l - lp + s_.nThetaEven) % s_.nThetaEven;

const int k_start = kl % s_.nZeta;
const int k_end = std::min(s_.nZeta, kl_end - l * s_.nZeta);
const int kl_backup = kl;

for (int k = k_start; k < k_end; ++k, ++kl) {
const int delta_k = (k - kp + s_.nZeta) % s_.nZeta;
ga1_buf[k] =
sg_.guu[klp - tp_.ztMin] * tanu[delta_l] * tanu[delta_l] +
sg_.guv[klp - tp_.ztMin] * tanu[delta_l] * tanv[delta_k] +
sg_.gvv[klp - tp_.ztMin] * tanv[delta_k] * tanv[delta_k];

ga2_buf[k] =
sg_.auu[klp - tp_.ztMin] * tanu[delta_l] * tanu[delta_l] +
sg_.auv[klp - tp_.ztMin] * tanu[delta_l] * tanv[delta_k] +
sg_.avv[klp - tp_.ztMin] * tanv[delta_k] * tanv[delta_k];
}

for (int k = k_start; k < k_end; ++k, ++kl) {
ga2_buf[k] /= ga1_buf[k];
ga1_buf[k] = 1.0 / sqrt(ga1_buf[k]);
}

kl = kl_backup;
for (int k = k_start; k < k_end; ++k, ++kl) {
ftemp_buf[k] =
1.0 /
(gsave[kl] - 2 * (xper * sg_.rcosuv[kl] + yper * sg_.rsinuv[kl]));
htemp_buf[k] = sqrt(ftemp_buf[k]);
}

kl = kl_backup;
for (int k = k_start; k < k_end; ++k, ++kl) {
const int ip = ip_idx_base + kl;

greenp[ip] +=
twopidivnfp * (htemp_buf[k] * ftemp_buf[k] *
(sg_.rcosuv[kl] * sxsave +
sg_.rsinuv[kl] * sysave + dsave[kl]) -
ga1_buf[k] * ga2_buf[k]);
const double g = twopidivnfp * (htemp_buf[k] - ga1_buf[k]);
gstore[kl] += bexni * g;
}
}
};

do_loop(  0,   klp);

do_loop(  klp + 1,   theta_by_nzeta);

for (int p = 1; p < s_.nfp; ++p) {
double xper = xp * sg_.cos_per[p] - yp * sg_.sin_per[p];
double yper = xp * sg_.sin_per[p] + yp * sg_.cos_per[p];
double sxsave =
(sg_.snr[klp - tp_.ztMin] * xper - sg_.snv[klp - tp_.ztMin] * yper) /
sg_.r1b[klp];
double sysave =
(sg_.snr[klp - tp_.ztMin] * yper + sg_.snv[klp - tp_.ztMin] * xper) /
sg_.r1b[klp];

for (int kl = 0; kl < theta_by_nzeta; ++kl) {
double ftemp =
1.0 /
(gsave[kl] - 2 * (xper * sg_.rcosuv[kl] + yper * sg_.rsinuv[kl]));
double htemp = sqrt(ftemp);

greenp[ip_idx_base + kl] +=
twopidivnfp * htemp * ftemp *
(sg_.rcosuv[kl] * sxsave + sg_.rsinuv[kl] * sysave + dsave[kl]);
const double g = twopidivnfp * htemp;
gstore[kl] += bexni * g;
}
}
}
}

void RegularizedIntegrals::updateAxisymmetric(const Eigen::VectorXd& bDotN) {

const int numLocal = tp_.ztMax - tp_.ztMin;
const int nThetaEven = s_.nThetaEven;

absl::c_fill_n(greenp, numLocal * nThetaEven, 0);
absl::c_fill_n(gstore, nThetaEven, 0);

const double toroidal_measure = 2.0 * M_PI / nvper_;

for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
const int ip_idx_base = (klp - tp_.ztMin) * nThetaEven;
const int klpRel = klp - tp_.ztMin;
const int lp = klp;

const double bexni = bDotN[klpRel] * s_.wInt[lp];

const double xp = sg_.rcosuv[klp];
const double yp = sg_.rsinuv[klp];

for (int kl = 0; kl < nThetaEven; ++kl) {
gsave[kl] = sg_.rzb2[klp] + sg_.rzb2[kl] - 2 * sg_.z1b[kl] * sg_.z1b[klp];
dsave[kl] = sg_.drv[klpRel] + sg_.z1b[kl] * sg_.snz[klpRel];
}

for (int p = 0; p < nvper_; ++p) {
const double cosper = std::cos(toroidal_measure * p);
const double sinper = std::sin(toroidal_measure * p);

const double xper = xp * cosper - yp * sinper;
const double yper = xp * sinper + yp * cosper;

const double sxsave =
(sg_.snr[klpRel] * xper - sg_.snv[klpRel] * yper) / sg_.r1b[klp];
const double sysave =
(sg_.snr[klpRel] * yper + sg_.snv[klpRel] * xper) / sg_.r1b[klp];

const double tanv_p = tanv_per_[p];

for (int kl = 0; kl < nThetaEven; ++kl) {

if (p == 0 && kl == klp) {
continue;
}

const int delta_l = (kl - lp + nThetaEven) % nThetaEven;

double ga1 = sg_.guu[klpRel] * tanu[delta_l] * tanu[delta_l] +
sg_.guv[klpRel] * tanu[delta_l] * tanv_p +
sg_.gvv[klpRel] * tanv_p * tanv_p;
double ga2 = sg_.auu[klpRel] * tanu[delta_l] * tanu[delta_l] +
sg_.auv[klpRel] * tanu[delta_l] * tanv_p +
sg_.avv[klpRel] * tanv_p * tanv_p;
ga2 /= ga1;
ga1 = 1.0 / std::sqrt(ga1);

const double ftemp =
1.0 /
(gsave[kl] - 2 * (xper * sg_.rcosuv[kl] + yper * sg_.rsinuv[kl]));
const double htemp = std::sqrt(ftemp);

greenp[ip_idx_base + kl] +=
toroidal_measure * (htemp * ftemp *
(sg_.rcosuv[kl] * sxsave +
sg_.rsinuv[kl] * sysave + dsave[kl]) -
ga1 * ga2);
gstore[kl] += bexni * toroidal_measure * (htemp - ga1);
}
}
}
}

}

// source: vmecpp/free_boundary/singular_integrals/singular_integrals.cc

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <vector>

namespace vmecpp {

namespace {

constexpr double kMinBoundaryLogDecay = 17.0 * std::numbers::ln10;

constexpr int kMinTail = 8;
constexpr int kMaxTail = 4096;

double ComputeT0(double a, double b2, double c) {
const double A = a + b2 + c;
const double root = 2.0 * std::sqrt(a * c);
const double determinant = (root - b2) * (root + b2);
const double hi = 2.0 * c + b2;
const double lo = 2.0 * a + b2;
const double sqrt_ca = 2.0 * std::sqrt(c * A);
const double sqrt_aa = 2.0 * std::sqrt(a * A);
const double numerator =
hi >= 0.0 ? sqrt_ca + hi : determinant / (sqrt_ca - hi);
const double denominator =
lo >= 0.0 ? determinant / (sqrt_aa + lo) : sqrt_aa - lo;
return std::log(numerator / denominator) / std::sqrt(A);
}

double EvaluateCmnPolynomial(int m, int n, double t) {
const int k = std::abs(m - n);
const int mu = std::min(m, n);
const double x = 1.0 - 2.0 * t * t;
double p_previous = 0.0;
double p = 1.0;
if (mu >= 1) {
p_previous = 1.0;
p = 0.5 * ((k + 2) * x + k);
for (int j = 1; j < mu; ++j) {
const double two_j_k = 2.0 * j + k;
const double c_next = 2.0 * (j + 1) * (j + k + 1) * two_j_k;
const double c_this =
(two_j_k + 1.0) * ((two_j_k + 2.0) * two_j_k * x + k * k);
const double c_previous = 2.0 * (j + k) * j * (two_j_k + 2.0);
const double p_next = (c_this * p - c_previous * p_previous) / c_next;
p_previous = p;
p = p_next;
}
}
const double sign = (std::max(0, n - m) % 2 == 0) ? 1.0 : -1.0;
return sign * std::pow(t, k) * p;
}

void ComputeChebyshevMoments(const Eigen::VectorXd& a,
const Eigen::VectorXd& b2,
const Eigen::VectorXd& c, double b2_sign,
int num_points, int kL,
ChebyshevMomentWorkspace& m_workspace,
std::vector<Eigen::VectorXd>& m_moments) {
using Batch = Eigen::Array<double, kChebyshevMomentBatch, 1>;

const std::vector<double>& factor_sub2 = m_workspace.factor_sub2;
const std::vector<double>& factor_sub1 = m_workspace.factor_sub1;
const std::vector<double>& factor_diag = m_workspace.factor_diag;
const std::vector<double>& factor_sup1 = m_workspace.factor_sup1;
const std::vector<double>& factor_sup2 = m_workspace.factor_sup2;
const std::vector<double>& factor_rhs = m_workspace.factor_rhs;
auto& upper1 = m_workspace.upper1;
auto& upper2 = m_workspace.upper2;
auto& moments = m_workspace.moments;

upper1.leftCols(2).setZero();
upper2.leftCols(2).setZero();

for (int first = 0; first < num_points; first += kChebyshevMomentBatch) {
Batch A;
Batch d;
Batch half_a_plus_b;
Batch sqrt_q_plus;
Batch sqrt_q_minus;
int tail = kMinTail;
for (int point = 0; point < kChebyshevMomentBatch; ++point) {

const int kl = std::min(first + point, num_points - 1);
const double b2_kl = b2_sign * b2[kl];
A[point] = a[kl] + b2_kl + c[kl];
d[point] = c[kl] - a[kl];
half_a_plus_b[point] = 0.5 * A[point] + (a[kl] - b2_kl + c[kl]);
sqrt_q_plus[point] = 2.0 * std::sqrt(c[kl]);
sqrt_q_minus[point] = 2.0 * std::sqrt(a[kl]);

const double m0 = ComputeT0(a[kl], b2_kl, c[kl]);
moments(point, 0) = m0;

moments(point, 1) =
(sqrt_q_plus[point] - sqrt_q_minus[point] - d[point] * m0) / A[point];

const double semi_major = 0.5 *
(sqrt_q_plus[point] + sqrt_q_minus[point]) /
std::sqrt(A[point]);
const double semi_minor = std::sqrt(
std::max(2.0 * std::sqrt(a[kl] * c[kl]) - b2_kl, 0.0) / A[point]);
const double log_rho = std::log(semi_major + semi_minor);
int tail_kl = kMaxTail;
if (log_rho > 0.0) {
tail_kl = static_cast<int>(
std::min(static_cast<double>(kMaxTail),
std::ceil(kMinBoundaryLogDecay / log_rho)));
}
tail = std::max(tail, tail_kl);
}

const int k_top = kL + tail;
for (int k = 2; k <= k_top; ++k) {
const double sign = (k % 2 == 0) ? 1.0 : -1.0;
const Batch sub2 = A * factor_sub2[k];
const Batch sub1 = d * factor_sub1[k] - upper1.col(k - 2) * sub2;
const Batch inverse_pivot =
(half_a_plus_b - A * factor_diag[k] - upper2.col(k - 2) * sub2 -
upper1.col(k - 1) * sub1)
.inverse();
upper1.col(k) =
(d * factor_sup1[k] - upper2.col(k - 1) * sub1) * inverse_pivot;
upper2.col(k) = A * factor_sup2[k] * inverse_pivot;
moments.col(k) = (-(sqrt_q_plus + sign * sqrt_q_minus) * factor_rhs[k] -
moments.col(k - 2) * sub2 - moments.col(k - 1) * sub1) *
inverse_pivot;
}
for (int k = k_top + 1; k <= k_top + 2; ++k) {
const double sign = (k % 2 == 0) ? 1.0 : -1.0;
moments.col(k) =
-(sqrt_q_plus.inverse() + sign * sqrt_q_minus.inverse()) *
factor_rhs[k];
}
for (int k = k_top; k >= 2; --k) {
moments.col(k) -= upper1.col(k) * moments.col(k + 1) +
upper2.col(k) * moments.col(k + 2);
}

const int num_in_batch =
std::min(kChebyshevMomentBatch, num_points - first);
for (int k = 0; k <= kL; ++k) {
m_moments[k].segment(first, num_in_batch) =
moments.col(k).head(num_in_batch);
}
}
}

}

SingularIntegrals::SingularIntegrals(const Sizes* s,
const FourierBasisFastToroidal* fb,
const TangentialPartitioning* tp,
const SurfaceGeometry* sg, int nf, int mf)
: s_(*s), fb_(*fb), tp_(*tp), sg_(*sg), nf(nf), mf(mf) {
numSC = mf * (nf + 1);
numCS = (mf + 1) * nf;
nzLen = numSC + numCS;

chebyshev_coefficients.resize((1 + nf + mf) * (nf + 1) * (mf + 1));

int numLocal = tp_.ztMax - tp_.ztMin;

ap.resize(numLocal);
am.resize(numLocal);
d.resize(numLocal);
sqrtc2.resize(numLocal);
sqrta2.resize(numLocal);
delta4.resize(numLocal);

Ap.resize(numLocal);
Am.resize(numLocal);
D.resize(numLocal);
R1p.resize(numLocal);
R1m.resize(numLocal);
R0p.resize(numLocal);
R0m.resize(numLocal);
Ra1p.resize(numLocal);
Ra1m.resize(numLocal);

chebyshev_moments_p.resize(mf + nf + 1);
chebyshev_moments_m.resize(mf + nf + 1);
chebyshev_s_moments_p.resize(mf + nf + 1);
chebyshev_s_moments_m.resize(mf + nf + 1);
for (int k = 0; k < mf + nf + 1; ++k) {
chebyshev_moments_p[k].resize(numLocal);
chebyshev_moments_m[k].resize(numLocal);
chebyshev_s_moments_p[k].resize(numLocal);
chebyshev_s_moments_m[k].resize(numLocal);
}

const int k_max = mf + nf + kMaxTail;
moment_workspace_.factor_sub2.resize(k_max + 3);
moment_workspace_.factor_sub1.resize(k_max + 3);
moment_workspace_.factor_diag.resize(k_max + 3);
moment_workspace_.factor_sup1.resize(k_max + 3);
moment_workspace_.factor_sup2.resize(k_max + 3);
moment_workspace_.factor_rhs.resize(k_max + 3);
for (int k = 2; k < k_max + 3; ++k) {
const double kd = k;
moment_workspace_.factor_sub2[k] = (kd - 2.0) / (4.0 * (kd - 1.0));
moment_workspace_.factor_sub1[k] = (2.0 * kd - 3.0) / (2.0 * (kd - 1.0));
moment_workspace_.factor_diag[k] = 1.0 / (2.0 * (kd * kd - 1.0));
moment_workspace_.factor_sup1[k] = (2.0 * kd + 3.0) / (2.0 * (kd + 1.0));
moment_workspace_.factor_sup2[k] = (kd + 2.0) / (4.0 * (kd + 1.0));
moment_workspace_.factor_rhs[k] = 1.0 / (kd * kd - 1.0);
}
moment_workspace_.upper1.resize(kChebyshevMomentBatch, k_max + 3);
moment_workspace_.upper2.resize(kChebyshevMomentBatch, k_max + 3);
moment_workspace_.moments.resize(kChebyshevMomentBatch, k_max + 3);

const int mnfull = (2 * nf + 1) * (mf + 1);
bvec_sin.setZero(mnfull);
grpmn_sin.setZero(mnfull * numLocal);
if (s->lasym) {
bvec_cos.setZero(mnfull);
grpmn_cos.setZero(mnfull * numLocal);
}

computeCoefficients();
}

void SingularIntegrals::computeCoefficients() {
const int kL = mf + nf;
const int num_nodes = kL + 1;

std::vector<double> theta(num_nodes);
for (int i = 0; i < num_nodes; ++i) {
theta[i] = (i + 0.5) * std::numbers::pi / num_nodes;
}

std::vector<double> cmn_values(static_cast<std::size_t>(nf + 1) * (mf + 1) *
num_nodes);
for (int n = 0; n < nf + 1; ++n) {
for (int m = 0; m < mf + 1; ++m) {
for (int i = 0; i < num_nodes; ++i) {
cmn_values[(n * (mf + 1) + m) * num_nodes + i] =
EvaluateCmnPolynomial(m, n, std::cos(theta[i]));
}
}
}

chebyshev_coefficients.setZero();
std::vector<double> p(num_nodes);
for (int n = 0; n < nf + 1; ++n) {
for (int m = 0; m < mf + 1; ++m) {

const int n_m_ = (n * (mf + 1) + m) * num_nodes;
const int n1m_ = ((n - 1) * (mf + 1) + m) * num_nodes;
const int n_m1 = (n * (mf + 1) + (m - 1)) * num_nodes;
const int n1m1 = ((n - 1) * (mf + 1) + (m - 1)) * num_nodes;
for (int i = 0; i < num_nodes; ++i) {
if (m == 0 && n == 0) {
p[i] = cmn_values[n_m_ + i];
} else if (m == 0 && n > 0) {
p[i] = (cmn_values[n_m_ + i] + cmn_values[n1m_ + i]) / 2;
} else if (m > 0 && n == 0) {
p[i] = (cmn_values[n_m_ + i] + cmn_values[n_m1 + i]) / 2;
} else {
p[i] = (cmn_values[n_m_ + i] + cmn_values[n1m_ + i] +
cmn_values[n_m1 + i] + cmn_values[n1m1 + i]) /
2;
}
}

for (int k = 0; k <= m + n; ++k) {
double sum = 0.0;
for (int i = 0; i < num_nodes; ++i) {
sum += p[i] * std::cos(k * theta[i]);
}
const int knm = (k * (nf + 1) + n) * (mf + 1) + m;
chebyshev_coefficients[knm] = (k == 0 ? 1.0 : 2.0) * sum / num_nodes;
}
}
}
}

void SingularIntegrals::update(const Eigen::VectorXd& bDotN, bool fullUpdate) {
#ifdef _OPENMP
#pragma omp barrier
#endif

prepareUpdate(sg_.guu, sg_.guv, sg_.gvv, sg_.auu, sg_.auv, sg_.avv,
fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif

performUpdate(bDotN, fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif
}

void SingularIntegrals::prepareUpdate(
const Eigen::VectorXd& a, const Eigen::VectorXd& b2,
const Eigen::VectorXd& c, const Eigen::VectorXd& A,
const Eigen::VectorXd& B2, const Eigen::VectorXd& C, bool fullUpdate) {
const int kL = mf + nf;
int numLocal = tp_.ztMax - tp_.ztMin;

ComputeChebyshevMoments(a, b2, c,  1.0, numLocal, kL,
moment_workspace_, chebyshev_moments_p);
ComputeChebyshevMoments(a, b2, c,  -1.0, numLocal, kL,
moment_workspace_, chebyshev_moments_m);

for (int kl = 0; kl < numLocal; ++kl) {

ap[kl] = a[kl] + b2[kl] + c[kl];
am[kl] = a[kl] - b2[kl] + c[kl];
d[kl] = c[kl] - a[kl];
sqrtc2[kl] = 2.0 * sqrt(c[kl]);
sqrta2[kl] = 2.0 * sqrt(a[kl]);

if (fullUpdate) {
delta4[kl] = ap[kl] * am[kl] - d[kl] * d[kl];

Ap[kl] = A[kl] + B2[kl] + C[kl];
Am[kl] = A[kl] - B2[kl] + C[kl];
D[kl] = C[kl] - A[kl];

R1p[kl] = (Ap[kl] * (delta4[kl] - d[kl] * d[kl]) / ap[kl] -
Am[kl] * ap[kl] + 2 * D[kl] * d[kl]) /
delta4[kl];
R1m[kl] = (Am[kl] * (delta4[kl] - d[kl] * d[kl]) / am[kl] -
Ap[kl] * am[kl] + 2 * D[kl] * d[kl]) /
delta4[kl];
R0p[kl] = (-Ap[kl] * am[kl] * d[kl] / ap[kl] - Am[kl] * d[kl] +
2 * D[kl] * am[kl]) /
delta4[kl];
R0m[kl] = (-Am[kl] * ap[kl] * d[kl] / am[kl] - Ap[kl] * d[kl] +
2 * D[kl] * ap[kl]) /
delta4[kl];
Ra1p[kl] = Ap[kl] / ap[kl];
Ra1m[kl] = Am[kl] / am[kl];

const auto s_moments = [&](const std::vector<Eigen::VectorXd>& moments,
double R0, double R1, double Ra1,
std::vector<Eigen::VectorXd>& m_s_moments) {
const double at_plus = -(R0 + R1) / sqrtc2[kl];
const double at_minus = (R0 - R1) / sqrta2[kl];
double n_km2 = 0.0;
double n_km1 = 0.0;
for (int k = 0; k <= kL; ++k) {
const double n_k = (k == 0 ? 1.0 : 2.0) * moments[k][kl] + n_km2;
m_s_moments[k][kl] = R1 * k * 0.5 * (n_k + n_km2) +
Ra1 * moments[k][kl] + R0 * k * n_km1 + at_plus +
(k % 2 == 0 ? at_minus : -at_minus);
n_km2 = n_km1;
n_km1 = n_k;
}
};
s_moments(chebyshev_moments_p, R0p[kl], R1p[kl], Ra1p[kl],
chebyshev_s_moments_p);
s_moments(chebyshev_moments_m, R0m[kl], R1m[kl], Ra1m[kl],
chebyshev_s_moments_m);
}
}
}

void SingularIntegrals::performUpdate(const Eigen::VectorXd& bDotN,
bool fullUpdate) {
const int numLocal = tp_.ztMax - tp_.ztMin;

bvec_sin.setZero();
if (s_.lasym) {
bvec_cos.setZero();
}

if (fullUpdate) {
grpmn_sin.setZero();
if (s_.lasym) {
grpmn_cos.setZero();
}
}

for (int order = 0; order < 1 + nf + mf; ++order) {
for (int n = 0; n < nf + 1; ++n) {
for (int m = 0; m < mf + 1; ++m) {
const int idx_m_posn = (nf + n) * (mf + 1) + m;
const int idx_m_negn = (nf - n) * (mf + 1) + m;

if (order > m + n) {
continue;
}

const int idx_knm = (order * (nf + 1) + n) * (mf + 1) + m;
const double gamma_factor =
chebyshev_coefficients[idx_knm] / (fb_.mscale[m] * fb_.nscale[n]);

if (n == 0 || m == 0) {

for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
const int l = kl / s_.nZeta;
const int k = kl % s_.nZeta;
const int klRel = kl - tp_.ztMin;

const int lr = (l < s_.nThetaReduced) ? l : (s_.nThetaEven - l);
const double sgnmu = (l < s_.nThetaReduced) ? 1.0 : -1.0;
const int idx_lm = lr * (s_.mnyq2 + 1) + m;
const int idx_nk = n * s_.nZeta + k;

const double sinp = (sgnmu * fb_.sinmu[idx_lm] * fb_.cosnv[idx_nk] -
fb_.cosmu[idx_lm] * fb_.sinnv[idx_nk]) *
gamma_factor;

bvec_sin[idx_m_posn] += (chebyshev_moments_p[order][klRel] +
chebyshev_moments_m[order][klRel]) *
bDotN[klRel] * s_.wInt[l] * sinp;
if (fullUpdate) {
grpmn_sin[idx_m_posn * numLocal + klRel] +=
(chebyshev_s_moments_p[order][klRel] +
chebyshev_s_moments_m[order][klRel]) *
sinp;
}

if (s_.lasym) {

const double cosp =
(fb_.cosmu[idx_lm] * fb_.cosnv[idx_nk] +
sgnmu * fb_.sinmu[idx_lm] * fb_.sinnv[idx_nk]) *
gamma_factor;

bvec_cos[idx_m_posn] += (chebyshev_moments_p[order][klRel] +
chebyshev_moments_m[order][klRel]) *
bDotN[klRel] * s_.wInt[l] * cosp;
if (fullUpdate) {
grpmn_cos[idx_m_posn * numLocal + klRel] +=
(chebyshev_s_moments_p[order][klRel] +
chebyshev_s_moments_m[order][klRel]) *
cosp;
}
}
}

} else {

for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
const int l = kl / s_.nZeta;
int k = kl % s_.nZeta;

const int lr = (l < s_.nThetaReduced) ? l : (s_.nThetaEven - l);
const double sgnmu = (l < s_.nThetaReduced) ? 1.0 : -1.0;
const int idx_lm = lr * (s_.mnyq2 + 1) + m;
const int remaining = std::min(s_.nZeta - k, tp_.ztMax - kl);

const double coeff1 = sgnmu * fb_.sinmu[idx_lm] * gamma_factor;
const double coeff2 = fb_.cosmu[idx_lm] * gamma_factor;

std::array<double, 4> buf_m_posn{};
std::array<double, 4> buf_m_negn{};

int i = 0;
for (; i + 3 < remaining; i += 4, k += 4, kl += 4) {

const int klRel = kl - tp_.ztMin;
const int idx_nk = n * s_.nZeta + k;

const double c0 = bDotN[klRel + 0] * s_.wInt[l];
const double c1 = bDotN[klRel + 1] * s_.wInt[l];
const double c2 = bDotN[klRel + 2] * s_.wInt[l];
const double c3 = bDotN[klRel + 3] * s_.wInt[l];

const double sinp0 = coeff1 * fb_.cosnv[idx_nk + 0] -
coeff2 * fb_.sinnv[idx_nk + 0];
const double sinp1 = coeff1 * fb_.cosnv[idx_nk + 1] -
coeff2 * fb_.sinnv[idx_nk + 1];
const double sinp2 = coeff1 * fb_.cosnv[idx_nk + 2] -
coeff2 * fb_.sinnv[idx_nk + 2];
const double sinp3 = coeff1 * fb_.cosnv[idx_nk + 3] -
coeff2 * fb_.sinnv[idx_nk + 3];

buf_m_posn[0] +=
chebyshev_moments_m[order][klRel + 0] * c0 * sinp0;
buf_m_posn[1] +=
chebyshev_moments_m[order][klRel + 1] * c1 * sinp1;
buf_m_posn[2] +=
chebyshev_moments_m[order][klRel + 2] * c2 * sinp2;
buf_m_posn[3] +=
chebyshev_moments_m[order][klRel + 3] * c3 * sinp3;

const double sinm0 = coeff1 * fb_.cosnv[idx_nk + 0] +
coeff2 * fb_.sinnv[idx_nk + 0];
const double sinm1 = coeff1 * fb_.cosnv[idx_nk + 1] +
coeff2 * fb_.sinnv[idx_nk + 1];
const double sinm2 = coeff1 * fb_.cosnv[idx_nk + 2] +
coeff2 * fb_.sinnv[idx_nk + 2];
const double sinm3 = coeff1 * fb_.cosnv[idx_nk + 3] +
coeff2 * fb_.sinnv[idx_nk + 3];

buf_m_negn[0] +=
chebyshev_moments_p[order][klRel + 0] * c0 * sinm0;
buf_m_negn[1] +=
chebyshev_moments_p[order][klRel + 1] * c1 * sinm1;
buf_m_negn[2] +=
chebyshev_moments_p[order][klRel + 2] * c2 * sinm2;
buf_m_negn[3] +=
chebyshev_moments_p[order][klRel + 3] * c3 * sinm3;

if (fullUpdate) {
grpmn_sin[idx_m_posn * numLocal + klRel + 0] +=
chebyshev_s_moments_m[order][klRel + 0] * sinp0;
grpmn_sin[idx_m_posn * numLocal + klRel + 1] +=
chebyshev_s_moments_m[order][klRel + 1] * sinp1;
grpmn_sin[idx_m_posn * numLocal + klRel + 2] +=
chebyshev_s_moments_m[order][klRel + 2] * sinp2;
grpmn_sin[idx_m_posn * numLocal + klRel + 3] +=
chebyshev_s_moments_m[order][klRel + 3] * sinp3;

grpmn_sin[idx_m_negn * numLocal + klRel + 0] +=
chebyshev_s_moments_p[order][klRel + 0] * sinm0;
grpmn_sin[idx_m_negn * numLocal + klRel + 1] +=
chebyshev_s_moments_p[order][klRel + 1] * sinm1;
grpmn_sin[idx_m_negn * numLocal + klRel + 2] +=
chebyshev_s_moments_p[order][klRel + 2] * sinm2;
grpmn_sin[idx_m_negn * numLocal + klRel + 3] +=
chebyshev_s_moments_p[order][klRel + 3] * sinm3;
}
}

bvec_sin[idx_m_posn] +=
buf_m_posn[0] + buf_m_posn[1] + buf_m_posn[2] + buf_m_posn[3];
bvec_sin[idx_m_negn] +=
buf_m_negn[0] + buf_m_negn[1] + buf_m_negn[2] + buf_m_negn[3];

if (i != remaining) {
for (; i < remaining; ++i, ++k, ++kl) {

const int klRel = kl - tp_.ztMin;
const int idx_nk = n * s_.nZeta + k;

const double coeff1 = sgnmu * fb_.sinmu[idx_lm] *
fb_.cosnv[idx_nk] * gamma_factor;
const double coeff2 =
fb_.cosmu[idx_lm] * fb_.sinnv[idx_nk] * gamma_factor;

const double sinm = coeff1 + coeff2;

const double sinp = coeff1 - coeff2;

const double c = bDotN[klRel] * s_.wInt[l];
bvec_sin[idx_m_posn] +=
chebyshev_moments_m[order][klRel] * c * sinp;
bvec_sin[idx_m_negn] +=
chebyshev_moments_p[order][klRel] * c * sinm;

if (fullUpdate) {
grpmn_sin[idx_m_posn * numLocal + klRel] +=
chebyshev_s_moments_m[order][klRel] * sinp;
grpmn_sin[idx_m_negn * numLocal + klRel] +=
chebyshev_s_moments_p[order][klRel] * sinm;
}
}
}

--kl;
}

if (s_.lasym) {
for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
const int l = kl / s_.nZeta;
const int k = kl % s_.nZeta;
const int klRel = kl - tp_.ztMin;

const int lr = (l < s_.nThetaReduced) ? l : (s_.nThetaEven - l);
const double sgnmu = (l < s_.nThetaReduced) ? 1.0 : -1.0;
const int idx_lm = lr * (s_.mnyq2 + 1) + m;
const int idx_nk = n * s_.nZeta + k;

const double coeff1 =
fb_.cosmu[idx_lm] * fb_.cosnv[idx_nk] * gamma_factor;
const double coeff2 =
sgnmu * fb_.sinmu[idx_lm] * fb_.sinnv[idx_nk] * gamma_factor;

const double cosm = coeff1 - coeff2;

const double cosp = coeff1 + coeff2;

bvec_cos[idx_m_posn] += chebyshev_moments_m[order][klRel] *
bDotN[klRel] * s_.wInt[l] * cosp;
bvec_cos[idx_m_negn] += chebyshev_moments_p[order][klRel] *
bDotN[klRel] * s_.wInt[l] * cosm;
if (fullUpdate) {
grpmn_cos[idx_m_posn * numLocal + klRel] +=
chebyshev_s_moments_m[order][klRel] * cosp;
grpmn_cos[idx_m_negn * numLocal + klRel] +=
chebyshev_s_moments_p[order][klRel] * cosm;
}
}
}
}
}
}

}
}

}

// source: vmecpp/free_boundary/surface_geometry/surface_geometry.cc

namespace vmecpp {

SurfaceGeometry::SurfaceGeometry(const Sizes* s,
const FourierBasisFastToroidal* fb,
const TangentialPartitioning* tp)
: s_(*s), fb_(*fb), tp_(*tp) {
cos_per.resize(s_.nfp);
sin_per.resize(s_.nfp);

cos_phi.resize(s_.nZeta);
sin_phi.resize(s_.nZeta);

r1b.resize(s_.nThetaEven * s_.nZeta);
z1b.resize(s_.nThetaEven * s_.nZeta);
rcosuv.resize(s_.nThetaEven * s_.nZeta);
rsinuv.resize(s_.nThetaEven * s_.nZeta);
rzb2.resize(s_.nThetaEven * s_.nZeta);

int numLocal = tp_.ztMax - tp_.ztMin;

const int derivSize = s_.lasym ? s_.nZnT : numLocal;

rub.resize(derivSize);
rvb.resize(derivSize);
zub.resize(derivSize);
zvb.resize(derivSize);

ruu.resize(derivSize);
ruv.resize(derivSize);
rvv.resize(derivSize);
zuu.resize(derivSize);
zuv.resize(derivSize);
zvv.resize(derivSize);

snr.resize(numLocal);
snv.resize(numLocal);
snz.resize(numLocal);

guu.resize(numLocal);
guv.resize(numLocal);
gvv.resize(numLocal);

auu.resize(numLocal);
auv.resize(numLocal);
avv.resize(numLocal);

drv.resize(numLocal);

if (s_.lasym) {
r1b_asym.resize(s_.nZnT);
z1b_asym.resize(s_.nZnT);
rub_asym.resize(s_.nZnT);
rvb_asym.resize(s_.nZnT);
zub_asym.resize(s_.nZnT);
zvb_asym.resize(s_.nZnT);
ruu_asym.resize(s_.nZnT);
ruv_asym.resize(s_.nZnT);
rvv_asym.resize(s_.nZnT);
zuu_asym.resize(s_.nZnT);
zuv_asym.resize(s_.nZnT);
zvv_asym.resize(s_.nZnT);
}

computeConstants();
}

void SurfaceGeometry::computeConstants() {
double omega_per = 2.0 * M_PI / s_.nfp;
for (int p = 0; p < s_.nfp; ++p) {
double phi_per = omega_per * p;
cos_per[p] = cos(phi_per);
sin_per[p] = sin(phi_per);
}

double omega_phi = 2.0 * M_PI / (s_.nfp * s_.nZeta);
for (int k = 0; k < s_.nZeta; ++k) {
double phi = omega_phi * k;
cos_phi[k] = cos(phi);
sin_phi[k] = sin(phi);
}
}

void SurfaceGeometry::update(
const std::span<const double> rCC, const std::span<const double> rSS,
const std::span<const double> rSC, const std::span<const double> rCS,
const std::span<const double> zSC, const std::span<const double> zCS,
const std::span<const double> zCC, const std::span<const double> zSS,
int signOfJacobian, bool fullUpdate) {
#ifdef _OPENMP
#pragma omp barrier
#endif

inverseDFT(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif

derivedSurfaceQuantities(signOfJacobian, fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif
}

void SurfaceGeometry::inverseDFT(
const std::span<const double> rCC, const std::span<const double> rSS,
const std::span<const double> rSC, const std::span<const double> rCS,
const std::span<const double> zSC, const std::span<const double> zCS,
const std::span<const double> zCC, const std::span<const double> zSS,
bool fullUpdate) {

r1b.setZero();
z1b.setZero();
if (s_.lasym) {
r1b_asym.setZero();
z1b_asym.setZero();
}

const int derivOffset = s_.lasym ? 0 : tp_.ztMin;

rub.setZero();
rvb.setZero();
zub.setZero();
zvb.setZero();
if (s_.lasym) {
rub_asym.setZero();
rvb_asym.setZero();
zub_asym.setZero();
zvb_asym.setZero();
}

if (fullUpdate) {
ruu.setZero();
ruv.setZero();
rvv.setZero();
zuu.setZero();
zuv.setZero();
zvv.setZero();
if (s_.lasym) {
ruu_asym.setZero();
ruv_asym.setZero();
rvv_asym.setZero();
zuu_asym.setZero();
zuv_asym.setZero();
zvv_asym.setZero();
}
}

for (int n = 0; n < s_.ntor + 1; ++n) {

int nSq = n * s_.nfp * n * s_.nfp;

int lMin = tp_.ztMin / s_.nZeta;
int lMax = tp_.ztMax / s_.nZeta;

for (int l = 0; l < s_.nThetaReduced; ++l) {
double rmkcc = 0.0;
double rmkss = 0.0;
double zmksc = 0.0;
double zmkcs = 0.0;

double rmkcc_m = 0.0;
double rmkcc_mm = 0.0;
double rmkss_m = 0.0;
double rmkss_mm = 0.0;

double zmksc_m = 0.0;
double zmksc_mm = 0.0;
double zmkcs_m = 0.0;
double zmkcs_mm = 0.0;

double rmksc = 0.0;
double rmkcs = 0.0;
double zmkcc = 0.0;
double zmkss = 0.0;
double rmksc_m = 0.0;
double rmksc_mm = 0.0;
double rmkcs_m = 0.0;
double rmkcs_mm = 0.0;
double zmkcc_m = 0.0;
double zmkcc_mm = 0.0;
double zmkss_m = 0.0;
double zmkss_mm = 0.0;

for (int m = 0; m < s_.mpol; ++m) {
int idx_mn = n * s_.mpol + m;

int mSq = m * m;

double cosmu = fb_.cosmu[l * (s_.mnyq2 + 1) + m];
double sinmu = fb_.sinmu[l * (s_.mnyq2 + 1) + m];

rmkcc += rCC[idx_mn] * cosmu;
rmkss += rSS[idx_mn] * sinmu;
zmksc += zSC[idx_mn] * sinmu;
zmkcs += zCS[idx_mn] * cosmu;

if (s_.lasym) {
rmksc += rSC[idx_mn] * sinmu;
rmkcs += rCS[idx_mn] * cosmu;
zmkcc += zCC[idx_mn] * cosmu;
zmkss += zSS[idx_mn] * sinmu;
}

if (s_.lasym || (lMin <= l && l <= lMax)) {
double cosmum = fb_.cosmum[l * (s_.mnyq2 + 1) + m];
double sinmum = fb_.sinmum[l * (s_.mnyq2 + 1) + m];
double cosmumm = -mSq * fb_.cosmu[l * (s_.mnyq2 + 1) + m];
double sinmumm = -mSq * fb_.sinmu[l * (s_.mnyq2 + 1) + m];

rmkcc_m += rCC[idx_mn] * sinmum;
rmkcc_mm += rCC[idx_mn] * cosmumm;
rmkss_m += rSS[idx_mn] * cosmum;
rmkss_mm += rSS[idx_mn] * sinmumm;

zmksc_m += zSC[idx_mn] * cosmum;
zmksc_mm += zSC[idx_mn] * sinmumm;
zmkcs_m += zCS[idx_mn] * sinmum;
zmkcs_mm += zCS[idx_mn] * cosmumm;

if (s_.lasym) {
rmksc_m += rSC[idx_mn] * cosmum;
rmksc_mm += rSC[idx_mn] * sinmumm;
rmkcs_m += rCS[idx_mn] * sinmum;
rmkcs_mm += rCS[idx_mn] * cosmumm;
zmkcc_m += zCC[idx_mn] * sinmum;
zmkcc_mm += zCC[idx_mn] * cosmumm;
zmkss_m += zSS[idx_mn] * cosmum;
zmkss_mm += zSS[idx_mn] * sinmumm;
}
}
}

for (int k = 0; k < s_.nZeta; ++k) {
int idx_kl = l * s_.nZeta + k;

double cosnv = fb_.cosnv[n * s_.nZeta + k];
double sinnv = fb_.sinnv[n * s_.nZeta + k];

r1b[idx_kl] += rmkcc * cosnv + rmkss * sinnv;
z1b[idx_kl] += zmksc * cosnv + zmkcs * sinnv;

if (s_.lasym) {
r1b_asym[idx_kl] += rmksc * cosnv + rmkcs * sinnv;
z1b_asym[idx_kl] += zmkcc * cosnv + zmkss * sinnv;
}

if (s_.lasym || (tp_.ztMin <= idx_kl && idx_kl < tp_.ztMax)) {
double cosnvn = fb_.cosnvn[n * s_.nZeta + k];
double sinnvn = fb_.sinnvn[n * s_.nZeta + k];

rub[idx_kl - derivOffset] += rmkcc_m * cosnv + rmkss_m * sinnv;
rvb[idx_kl - derivOffset] += rmkcc * sinnvn + rmkss * cosnvn;
zub[idx_kl - derivOffset] += zmksc_m * cosnv + zmkcs_m * sinnv;
zvb[idx_kl - derivOffset] += zmksc * sinnvn + zmkcs * cosnvn;

if (s_.lasym) {
rub_asym[idx_kl - derivOffset] += rmksc_m * cosnv + rmkcs_m * sinnv;
rvb_asym[idx_kl - derivOffset] += rmksc * sinnvn + rmkcs * cosnvn;
zub_asym[idx_kl - derivOffset] += zmkcc_m * cosnv + zmkss_m * sinnv;
zvb_asym[idx_kl - derivOffset] += zmkcc * sinnvn + zmkss * cosnvn;
}

if (fullUpdate) {
double cosnvnn = -nSq * fb_.cosnv[n * s_.nZeta + k];
double sinnvnn = -nSq * fb_.sinnv[n * s_.nZeta + k];

ruu[idx_kl - derivOffset] += rmkcc_mm * cosnv + rmkss_mm * sinnv;
ruv[idx_kl - derivOffset] += rmkcc_m * sinnvn + rmkss_m * cosnvn;
rvv[idx_kl - derivOffset] += rmkcc * cosnvnn + rmkss * sinnvnn;
zuu[idx_kl - derivOffset] += zmksc_mm * cosnv + zmkcs_mm * sinnv;
zuv[idx_kl - derivOffset] += zmksc_m * sinnvn + zmkcs_m * cosnvn;
zvv[idx_kl - derivOffset] += zmksc * cosnvnn + zmkcs * sinnvnn;

if (s_.lasym) {
ruu_asym[idx_kl - derivOffset] +=
rmksc_mm * cosnv + rmkcs_mm * sinnv;
ruv_asym[idx_kl - derivOffset] +=
rmksc_m * sinnvn + rmkcs_m * cosnvn;
rvv_asym[idx_kl - derivOffset] +=
rmksc * cosnvnn + rmkcs * sinnvnn;
zuu_asym[idx_kl - derivOffset] +=
zmkcc_mm * cosnv + zmkss_mm * sinnv;
zuv_asym[idx_kl - derivOffset] +=
zmkcc_m * sinnvn + zmkss_m * cosnvn;
zvv_asym[idx_kl - derivOffset] +=
zmkcc * cosnvnn + zmkss * sinnvnn;
}
}
}
}
}
}

if (s_.lasym) {

for (int l = 1; l < s_.nThetaReduced - 1; ++l) {
const int lRev = (s_.nThetaEven - l) % s_.nThetaEven;
for (int k = 0; k < s_.nZeta; ++k) {
const int kRev = (s_.nZeta - k) % s_.nZeta;
const int kl = l * s_.nZeta + k;
const int klRev = lRev * s_.nZeta + kRev;

r1b[klRev] = r1b[kl] - r1b_asym[kl];
z1b[klRev] = -z1b[kl] + z1b_asym[kl];

rub[klRev] = -rub[kl] + rub_asym[kl];
rvb[klRev] = -rvb[kl] + rvb_asym[kl];
zub[klRev] = zub[kl] - zub_asym[kl];
zvb[klRev] = zvb[kl] - zvb_asym[kl];

if (fullUpdate) {
ruu[klRev] = ruu[kl] - ruu_asym[kl];
ruv[klRev] = ruv[kl] - ruv_asym[kl];
rvv[klRev] = rvv[kl] - rvv_asym[kl];
zuu[klRev] = -zuu[kl] + zuu_asym[kl];
zuv[klRev] = -zuv[kl] + zuv_asym[kl];
zvv[klRev] = -zvv[kl] + zvv_asym[kl];
}
}
}

for (int l = 0; l < s_.nThetaReduced; ++l) {
for (int k = 0; k < s_.nZeta; ++k) {
const int kl = l * s_.nZeta + k;

r1b[kl] += r1b_asym[kl];
z1b[kl] += z1b_asym[kl];

rub[kl] += rub_asym[kl];
rvb[kl] += rvb_asym[kl];
zub[kl] += zub_asym[kl];
zvb[kl] += zvb_asym[kl];

if (fullUpdate) {
ruu[kl] += ruu_asym[kl];
ruv[kl] += ruv_asym[kl];
rvv[kl] += rvv_asym[kl];
zuu[kl] += zuu_asym[kl];
zuv[kl] += zuv_asym[kl];
zvv[kl] += zvv_asym[kl];
}
}
}
}
}

void SurfaceGeometry::derivedSurfaceQuantities(int signOfJacobian,
bool fullUpdate) {

const int derivOffset = s_.lasym ? 0 : tp_.ztMin;
for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
const int o = kl - tp_.ztMin;
const int d = kl - derivOffset;

snr[o] = signOfJacobian * r1b[kl] * zub[d];
snv[o] = signOfJacobian * (rub[d] * zvb[d] - zub[d] * rvb[d]);
snz[o] = -signOfJacobian * r1b[kl] * rub[d];

guu[o] = rub[d] * rub[d] + zub[d] * zub[d];
guv[o] = 2.0 * (rub[d] * rvb[d] + zub[d] * zvb[d]) / s_.nfp;
gvv[o] = (rvb[d] * rvb[d] + r1b[kl] * r1b[kl] + zvb[d] * zvb[d]) /
(s_.nfp * s_.nfp);

if (fullUpdate) {

auu[o] = (ruu[d] * snr[o] + zuu[d] * snz[o]) / 2;
auv[o] = (ruv[d] * snr[o] + rub[d] * snv[o] + zuv[d] * snz[o]) / s_.nfp;
avv[o] = (rvb[d] * snv[o] +
((rvv[d] - r1b[kl]) * snr[o] + zvv[d] * snz[o]) / 2) /
(s_.nfp * s_.nfp);

drv[o] = -(r1b[kl] * snr[o] + z1b[kl] * snz[o]);
}
}

if (fullUpdate) {

for (int kl = 0; kl < s_.nZnT; ++kl) {
rzb2[kl] = r1b[kl] * r1b[kl] + z1b[kl] * z1b[kl];
}

if (!s_.lasym) {

for (int l = 1; l < s_.nThetaReduced - 1; ++l) {
int lRev = (s_.nThetaEven - l) % s_.nThetaEven;
for (int k = 0; k < s_.nZeta; ++k) {
int kRev = (s_.nZeta - k) % s_.nZeta;

int kl = l * s_.nZeta + k;
int klRev = lRev * s_.nZeta + kRev;

r1b[klRev] = r1b[kl];
z1b[klRev] = -z1b[kl];

rzb2[klRev] = rzb2[kl];
}
}
}

for (int kl = 0; kl < s_.nThetaEven * s_.nZeta; ++kl) {
int k = kl % s_.nZeta;
rcosuv[kl] = r1b[kl] * cos_phi[k];
rsinuv[kl] = r1b[kl] * sin_phi[k];
}
}
}

}

// source: vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.cc

namespace vmecpp {

TangentialPartitioning::TangentialPartitioning(int nZnT, int num_threads,
int thread_id)
: num_threads_(num_threads), thread_id_(thread_id) {
adjustPartitioning(nZnT);
}

void TangentialPartitioning::adjustPartitioning(int nZnT) {

int work_per_CPU = nZnT / num_threads_;
int work_remainder = nZnT % num_threads_;

ztMin = thread_id_ * work_per_CPU;
ztMax = (thread_id_ + 1) * work_per_CPU;
if (thread_id_ < work_remainder) {
ztMin += thread_id_;
ztMax += thread_id_ + 1;
} else {
ztMin += work_remainder;
ztMax += work_remainder;
}
}

int TangentialPartitioning::get_thread_id() const { return thread_id_; }

int TangentialPartitioning::get_num_threads() const { return num_threads_; }

}

// source: vmecpp/vmec/boundaries/boundaries.cc

// header: vmecpp/vmec/boundaries/boundaries.h

#ifndef VMECPP_VMEC_BOUNDARIES_BOUNDARIES_H_
#define VMECPP_VMEC_BOUNDARIES_BOUNDARIES_H_

#include <Eigen/Dense>

// header: vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h

#ifndef VMECPP_COMMON_FOURIER_BASIS_FAST_POLOIDAL_FOURIER_BASIS_FAST_POLOIDAL_H_
#define VMECPP_COMMON_FOURIER_BASIS_FAST_POLOIDAL_FOURIER_BASIS_FAST_POLOIDAL_H_

#endif

namespace vmecpp {

inline constexpr double kSpectrallyDenseBoundaryThreshold = 2.0;

class Boundaries {
public:
Boundaries(const Sizes* s, const FourierBasisFastPoloidal* t,
int sign_of_jacobian);

bool setupFromIndata(const VmecINDATA& id, bool verbose = true);
void ensureM1Constrained(double scaling_factor);

double ComputeSpectralWidth(int p = 4, int q = 1) const;

void RecomputeMagneticAxisToFixJacobianSign(int number_of_flux_surfaces,
int sign_of_jacobian);

Eigen::VectorXd raxis_c;
Eigen::VectorXd zaxis_s;
Eigen::VectorXd raxis_s;
Eigen::VectorXd zaxis_c;

Eigen::VectorXd rbcc;
Eigen::VectorXd rbss;
Eigen::VectorXd rbsc;
Eigen::VectorXd rbcs;

Eigen::VectorXd zbsc;
Eigen::VectorXd zbcs;
Eigen::VectorXd zbcc;
Eigen::VectorXd zbss;

private:
const Sizes& s_;
const FourierBasisFastPoloidal& t_;

int sign_of_jacobian_;

void parseToInternalArrays(const VmecINDATA& id, bool verbose = true);

bool checkSignOfJacobian();

void flipTheta();
};

}

#endif

#include <iostream>
#include <span>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"

// header: vmecpp/vmec/boundaries/guess_magnetic_axis.h

#ifndef VMECPP_VMEC_BOUNDARIES_GUESS_MAGNETIC_AXIS_H_
#define VMECPP_VMEC_BOUNDARIES_GUESS_MAGNETIC_AXIS_H_

#include <Eigen/Dense>
#include <vector>

namespace vmecpp {

struct RecomputeAxisWorkspace {

std::vector<double> r_axis;
std::vector<double> z_axis;

std::vector<std::vector<double> > r_lcfs;
std::vector<std::vector<double> > z_lcfs;
std::vector<std::vector<double> > d_r_d_theta_lcfs;
std::vector<std::vector<double> > d_z_d_theta_lcfs;

std::vector<std::vector<double> > r_half;
std::vector<std::vector<double> > z_half;
std::vector<std::vector<double> > d_r_d_theta_half;
std::vector<std::vector<double> > d_z_d_theta_half;
std::vector<std::vector<double> > d_r_d_s_half;
std::vector<std::vector<double> > d_z_d_s_half;

std::vector<std::vector<double> > tau0;
std::vector<std::vector<double> > tau;

std::vector<double> new_r_axis;
std::vector<double> new_z_axis;

std::vector<double> new_raxis_c;
std::vector<double> new_raxis_s;
std::vector<double> new_zaxis_s;
std::vector<double> new_zaxis_c;
};

RecomputeAxisWorkspace RecomputeMagneticAxisToFixJacobianSign(
int number_of_flux_surfaces, int sign_of_jacobian, const Sizes& s,
const FourierBasisFastPoloidal& t, const Eigen::VectorXd& rbcc,
const Eigen::VectorXd& rbss, const Eigen::VectorXd& rbsc,
const Eigen::VectorXd& rbcs, const Eigen::VectorXd& zbsc,
const Eigen::VectorXd& zbcs, const Eigen::VectorXd& zbcc,
const Eigen::VectorXd& zbss, const Eigen::VectorXd& raxis_c,
const Eigen::VectorXd& raxis_s, const Eigen::VectorXd& zaxis_s,
const Eigen::VectorXd& zaxis_c);

}

#endif

namespace vmecpp {

namespace {
std::span<const double> AsSpan(const Eigen::VectorXd& coefficients) {
return {coefficients.data(), static_cast<size_t>(coefficients.size())};
}
}

Boundaries::Boundaries(const Sizes* s, const FourierBasisFastPoloidal* t,
const int sign_of_jacobian)
: s_(*s), t_(*t), sign_of_jacobian_(sign_of_jacobian) {
raxis_c.resize(s_.ntor + 1);
zaxis_s.resize(s_.ntor + 1);
if (s_.lasym) {
raxis_s.resize(s_.ntor + 1);
zaxis_c.resize(s_.ntor + 1);
}

rbcc.resize(s_.mpol * (s_.ntor + 1));
zbsc.resize(s_.mpol * (s_.ntor + 1));
if (s_.lthreed) {
rbss.resize(s_.mpol * (s_.ntor + 1));
zbcs.resize(s_.mpol * (s_.ntor + 1));
}
if (s_.lasym) {
rbsc.resize(s_.mpol * (s_.ntor + 1));
zbcc.resize(s_.mpol * (s_.ntor + 1));
if (s_.lthreed) {
rbcs.resize(s_.mpol * (s_.ntor + 1));
zbss.resize(s_.mpol * (s_.ntor + 1));
}
}
}

bool Boundaries::setupFromIndata(const VmecINDATA& id, bool verbose) {
parseToInternalArrays(id, verbose);

bool haveToFlipTheta = checkSignOfJacobian();
if (haveToFlipTheta) {
if (verbose) {
std::cout << "need to flip theta definition of input boundary shape\n";
}
flipTheta();
}

ensureM1Constrained(0.5);

if (verbose && !id.lfreeb) {
const double spectral_width = ComputeSpectralWidth();
if (spectral_width > kSpectrallyDenseBoundaryThreshold) {
LOG(WARNING) << absl::StrFormat(
"Input boundary is spectrally dense (spectral width <M> = %.1f). "
"Runs with high spectral content may converge poorly or incorrectly "
"resolve the physics. Consider spectral condensation of the boundary "
"first, e.g. using simsopt.geo.surfacerzfourier.condense_spectrum",
spectral_width);
}
}

return haveToFlipTheta;
}

double Boundaries::ComputeSpectralWidth(const int p, const int q) const {

const Eigen::VectorXd unit_mscale = Eigen::VectorXd::Ones(s_.mpol);
const Eigen::VectorXd unit_nscale = Eigen::VectorXd::Ones(s_.ntor + 1);

const SurfaceFourierGeometry boundary = {.rmncc = AsSpan(rbcc),
.rmnss = AsSpan(rbss),
.rmnsc = AsSpan(rbsc),
.rmncs = AsSpan(rbcs),
.zmnsc = AsSpan(zbsc),
.zmncs = AsSpan(zbcs),
.zmncc = AsSpan(zbcc),
.zmnss = AsSpan(zbss)};

return SpectralWidth(boundary, s_, AsSpan(unit_mscale), AsSpan(unit_nscale),
p, q);
}

void Boundaries::parseToInternalArrays(const VmecINDATA& id, bool verbose) {

for (int n = 0; n < s_.ntor + 1; ++n) {
raxis_c[n] = id.raxis_c[n];
zaxis_s[n] = id.zaxis_s[n];
if (s_.lasym) {
raxis_s[n] = (*id.raxis_s)[n];
zaxis_c[n] = (*id.zaxis_c)[n];
}
}

double delta = 0.0;
if (s_.lasym) {

int m = 1;
int n = 0;

double r_test = 0.0;
double z_test = 0.0;
double r_test_asym = 0.0;
double z_test_asym = 0.0;
for (int nn = -s_.ntor; nn <= s_.ntor; ++nn) {
r_test += id.rbc(m, s_.ntor + nn);
z_test += id.zbs(m, s_.ntor + nn);
r_test_asym += (*id.rbs)(m, s_.ntor + nn);
z_test_asym += (*id.zbc)(m, s_.ntor + nn);
}
const double handedness = r_test * z_test - r_test_asym * z_test_asym;
const bool will_flip_theta = (handedness * sign_of_jacobian_ > 0.0);
const double sigma = -sign_of_jacobian_ * (will_flip_theta ? -1.0 : 1.0);
delta = atan2((*id.rbs)(m, s_.ntor + n) - sigma * (*id.zbc)(m, s_.ntor + n),
id.rbc(m, s_.ntor + n) + sigma * id.zbs(m, s_.ntor + n));

if (verbose && delta != 0.0) {
std::cout << "need to shift theta by delta = " << delta << "\n";

}
}

absl::c_fill_n(rbcc, s_.mpol * (s_.ntor + 1), 0);
absl::c_fill_n(zbsc, s_.mpol * (s_.ntor + 1), 0);
if (s_.lthreed) {
absl::c_fill_n(rbss, s_.mpol * (s_.ntor + 1), 0);
absl::c_fill_n(zbcs, s_.mpol * (s_.ntor + 1), 0);
}
if (s_.lasym) {
absl::c_fill_n(rbsc, s_.mpol * (s_.ntor + 1), 0);
absl::c_fill_n(zbcc, s_.mpol * (s_.ntor + 1), 0);
if (s_.lthreed) {
absl::c_fill_n(rbcs, s_.mpol * (s_.ntor + 1), 0);
absl::c_fill_n(zbss, s_.mpol * (s_.ntor + 1), 0);
}
}

for (int m = 0; m < s_.mpol; ++m) {
double cosMDelta = 1.0;
double sinMDelta = 0.0;
if (delta != 0.0) {
cosMDelta = cos(m * delta);
sinMDelta = sin(m * delta);
}

for (int n = -s_.ntor; n <= s_.ntor; ++n) {

int source_n = s_.ntor + n;

int target_n = abs(n);
double sign_n = signum(n);

double rbc;
double zbs;
if (!s_.lasym || delta == 0.0) {
rbc = id.rbc(m, source_n);
zbs = id.zbs(m, source_n);
} else {

rbc = id.rbc(m, source_n) * cosMDelta +
(*id.rbs)(m, source_n) * sinMDelta;
zbs = id.zbs(m, source_n) * cosMDelta -
(*id.zbc)(m, source_n) * sinMDelta;
}

const int idx_mn = m * (s_.ntor + 1) + target_n;
rbcc[idx_mn] += rbc;
if (m > 0) {
zbsc[idx_mn] += zbs;
}
if (s_.lthreed) {
if (m > 0) {
rbss[idx_mn] += sign_n * rbc;
}
zbcs[idx_mn] -= sign_n * zbs;
}

if (s_.lasym) {
double rbs;
double zbc;
if (delta == 0.0) {
rbs = (*id.rbs)(m, source_n);
zbc = (*id.zbc)(m, source_n);
} else {
rbs = (*id.rbs)(m, source_n) * cosMDelta -
id.rbc(m, source_n) * sinMDelta;
zbc = (*id.zbc)(m, source_n) * cosMDelta +
id.zbs(m, source_n) * sinMDelta;
}

if (m > 0) {
rbsc[idx_mn] += rbs;
}
zbcc[idx_mn] += zbc;
if (s_.lthreed) {
rbcs[idx_mn] -= sign_n * rbs;
if (m > 0) {
zbss[idx_mn] += sign_n * zbc;
}
}
}
}
}
}

bool Boundaries::checkSignOfJacobian() {

double rTest = 0.0;
double zTest = 0.0;
double rTestAsym = 0.0;
double zTestAsym = 0.0;
for (int n = 0; n < s_.ntor + 1; ++n) {
int m = 1;
int idx_mn = m * (s_.ntor + 1) + n;
rTest += rbcc[idx_mn];
zTest += zbsc[idx_mn];
if (s_.lasym) {
rTestAsym += rbsc[idx_mn];
zTestAsym += zbcc[idx_mn];
}
}

const double handedness = rTest * zTest - rTestAsym * zTestAsym;

return (handedness * sign_of_jacobian_ > 0.0);
}

void Boundaries::flipTheta() {
for (int m = 1; m < s_.mpol; ++m) {
for (int n = 0; n <= s_.ntor; ++n) {
int idx_mn = m * (s_.ntor + 1) + n;

int m_parity = ((m % 2 == 0) ? 1 : -1);

rbcc[idx_mn] *= m_parity;
zbsc[idx_mn] *= -m_parity;
if (s_.lthreed) {
rbss[idx_mn] *= -m_parity;
zbcs[idx_mn] *= m_parity;
}
if (s_.lasym) {
rbsc[idx_mn] *= -m_parity;
zbcc[idx_mn] *= m_parity;
if (s_.lthreed) {
rbcs[idx_mn] *= m_parity;
zbss[idx_mn] *= -m_parity;
}
}
}
}
}

void Boundaries::ensureM1Constrained(const double scaling_factor) {

const double sigma = -sign_of_jacobian_;
for (int n = 0; n <= s_.ntor; ++n) {
int m = 1;
int idx_mn = m * (s_.ntor + 1) + n;
if (s_.lthreed) {
double backup_rss = rbss[idx_mn];
rbss[idx_mn] = (backup_rss + sigma * zbcs[idx_mn]) * scaling_factor;
zbcs[idx_mn] = (sigma * backup_rss - zbcs[idx_mn]) * scaling_factor;
}
if (s_.lasym) {
double backup_rsc = rbsc[idx_mn];
rbsc[idx_mn] = (backup_rsc + sigma * zbcc[idx_mn]) * scaling_factor;
zbcc[idx_mn] = (sigma * backup_rsc - zbcc[idx_mn]) * scaling_factor;
}
}
}

void Boundaries::RecomputeMagneticAxisToFixJacobianSign(
const int number_of_flux_surfaces, const int sign_of_jacobian) {

RecomputeAxisWorkspace w = vmecpp::RecomputeMagneticAxisToFixJacobianSign(
number_of_flux_surfaces, sign_of_jacobian, s_, t_, rbcc, rbss, rbsc, rbcs,
zbsc, zbcs, zbcc, zbss, raxis_c, raxis_s, zaxis_s, zaxis_c);

for (int n = 0; n <= s_.ntor; ++n) {
raxis_c[n] = w.new_raxis_c[n];
zaxis_s[n] = w.new_zaxis_s[n];
if (s_.lasym) {
raxis_s[n] = w.new_raxis_s[n];
zaxis_c[n] = w.new_zaxis_c[n];
}
}
}

}

// source: vmecpp/vmec/boundaries/guess_magnetic_axis.cc

#include <algorithm>
#include <vector>

namespace vmecpp {

RecomputeAxisWorkspace RecomputeMagneticAxisToFixJacobianSign(
int number_of_flux_surfaces, int sign_of_jacobian, const Sizes& s,
const FourierBasisFastPoloidal& t, const Eigen::VectorXd& rbcc,
const Eigen::VectorXd& rbss, const Eigen::VectorXd& rbsc,
const Eigen::VectorXd& rbcs, const Eigen::VectorXd& zbsc,
const Eigen::VectorXd& zbcs, const Eigen::VectorXd& zbcc,
const Eigen::VectorXd& zbss, const Eigen::VectorXd& raxis_c,
const Eigen::VectorXd& raxis_s, const Eigen::VectorXd& zaxis_s,
const Eigen::VectorXd& zaxis_c) {
RecomputeAxisWorkspace w;

w.r_axis.resize(s.nZeta);
w.z_axis.resize(s.nZeta);

w.r_lcfs.resize(s.nZeta);
w.z_lcfs.resize(s.nZeta);
w.d_r_d_theta_lcfs.resize(s.nZeta);
w.d_z_d_theta_lcfs.resize(s.nZeta);
for (int k = 0; k < s.nZeta; ++k) {
w.r_lcfs[k].resize(s.nThetaEven);
w.z_lcfs[k].resize(s.nThetaEven);
w.d_r_d_theta_lcfs[k].resize(s.nThetaEven);
w.d_z_d_theta_lcfs[k].resize(s.nThetaEven);
}

w.r_half.resize(s.nZeta);
w.z_half.resize(s.nZeta);
w.d_r_d_theta_half.resize(s.nZeta);
w.d_z_d_theta_half.resize(s.nZeta);
w.d_r_d_s_half.resize(s.nZeta);
w.d_z_d_s_half.resize(s.nZeta);
for (int k = 0; k < s.nZeta; ++k) {
w.r_half[k].resize(s.nThetaEven);
w.z_half[k].resize(s.nThetaEven);
w.d_r_d_theta_half[k].resize(s.nThetaEven);
w.d_z_d_theta_half[k].resize(s.nThetaEven);
w.d_r_d_s_half[k].resize(s.nThetaEven);
w.d_z_d_s_half[k].resize(s.nThetaEven);
}

w.tau0.resize(s.nZeta);
w.tau.resize(s.nZeta);
for (int k = 0; k < s.nZeta; ++k) {
w.tau0[k].resize(s.nThetaEven);
w.tau[k].resize(s.nThetaEven);
}

w.new_r_axis.resize(s.nZeta);
w.new_z_axis.resize(s.nZeta);

w.new_raxis_c.resize(s.ntor + 1);
w.new_zaxis_s.resize(s.ntor + 1);
if (s.lasym) {
w.new_raxis_s.resize(s.ntor + 1);
w.new_zaxis_c.resize(s.ntor + 1);
}

static constexpr int kNumberOfGridPoints = 61;

const int ns12 = (number_of_flux_surfaces + 1) / 2 - 1;

const double delta_s =
(number_of_flux_surfaces - 1 - ns12) / (number_of_flux_surfaces - 1.0);

const double sqrtSF12 = std::sqrt(ns12 / (number_of_flux_surfaces - 1.0));

std::vector<std::vector<double> > rcc_half(s.mpol);
std::vector<std::vector<double> > rss_half;
std::vector<std::vector<double> > rsc_half;
std::vector<std::vector<double> > rcs_half;
std::vector<std::vector<double> > zsc_half(s.mpol);
std::vector<std::vector<double> > zcs_half;
std::vector<std::vector<double> > zcc_half;
std::vector<std::vector<double> > zss_half;
if (s.lthreed) {
rss_half.resize(s.mpol);
zcs_half.resize(s.mpol);
}
if (s.lasym) {
rsc_half.resize(s.mpol);
zcc_half.resize(s.mpol);
if (s.lthreed) {
rcs_half.resize(s.mpol);
zss_half.resize(s.mpol);
}
}

const double scalingFactor = 1.0;
const double sigma = -sign_of_jacobian;
std::vector<std::vector<double> > rss_boundary;
std::vector<std::vector<double> > zcs_boundary;
std::vector<std::vector<double> > rsc_boundary;
std::vector<std::vector<double> > zcc_boundary;
if (s.lthreed) {
rss_boundary.resize(s.mpol);
zcs_boundary.resize(s.mpol);
for (int m = 0; m < s.mpol; ++m) {
rss_boundary[m].resize(s.ntor + 1);
zcs_boundary[m].resize(s.ntor + 1);
for (int n = 0; n <= s.ntor; ++n) {
int idx_mn = m * (s.ntor + 1) + n;
if (m == 1) {
rss_boundary[m][n] =
(rbss[idx_mn] + sigma * zbcs[idx_mn]) * scalingFactor;
zcs_boundary[m][n] =
(sigma * rbss[idx_mn] - zbcs[idx_mn]) * scalingFactor;
} else {
rss_boundary[m][n] = rbss[idx_mn];
zcs_boundary[m][n] = zbcs[idx_mn];
}
}
}
}
if (s.lasym) {
rsc_boundary.resize(s.mpol);
zcc_boundary.resize(s.mpol);
for (int m = 0; m < s.mpol; ++m) {
rsc_boundary[m].resize(s.ntor + 1);
zcc_boundary[m].resize(s.ntor + 1);
for (int n = 0; n <= s.ntor; ++n) {
int idx_mn = m * (s.ntor + 1) + n;
if (m == 1) {
rsc_boundary[m][n] =
(rbsc[idx_mn] + sigma * zbcc[idx_mn]) * scalingFactor;
zcc_boundary[m][n] =
(sigma * rbsc[idx_mn] - zbcc[idx_mn]) * scalingFactor;
} else {
rsc_boundary[m][n] = rbsc[idx_mn];
zcc_boundary[m][n] = zbcc[idx_mn];
}
}
}
}

for (int m = 0; m < s.mpol; ++m) {
rcc_half[m].resize(s.ntor + 1);
zsc_half[m].resize(s.ntor + 1);
if (s.lthreed) {
rss_half[m].resize(s.ntor + 1);
zcs_half[m].resize(s.ntor + 1);
}
if (s.lasym) {
rsc_half[m].resize(s.ntor + 1);
zcc_half[m].resize(s.ntor + 1);
if (s.lthreed) {
rcs_half[m].resize(s.ntor + 1);
zss_half[m].resize(s.ntor + 1);
}
}

for (int n = 0; n <= s.ntor; ++n) {
const int idx_mn = m * (s.ntor + 1) + n;

if (m == 0) {

const double interpolation_weight = sqrtSF12 * sqrtSF12;

rcc_half[m][n] = (interpolation_weight * rbcc[idx_mn] +
(1.0 - interpolation_weight) * raxis_c[n]);

zsc_half[m][n] = interpolation_weight * zbsc[idx_mn];
if (s.lthreed) {

rss_half[m][n] = interpolation_weight * rss_boundary[m][n];
zcs_half[m][n] = (interpolation_weight * zcs_boundary[m][n] -
(1.0 - interpolation_weight) * zaxis_s[n]);
}
if (s.lasym) {

rsc_half[m][n] = interpolation_weight * rsc_boundary[m][n];
zcc_half[m][n] = (interpolation_weight * zcc_boundary[m][n] +
(1.0 - interpolation_weight) * zaxis_c[n]);
if (s.lthreed) {
rcs_half[m][n] = (interpolation_weight * rbcs[idx_mn] -
(1.0 - interpolation_weight) * raxis_s[n]);

zss_half[m][n] = interpolation_weight * zbss[idx_mn];
}
}
} else {

const double interpolation_weight = std::pow(sqrtSF12, m);

rcc_half[m][n] = interpolation_weight * rbcc[idx_mn];
zsc_half[m][n] = interpolation_weight * zbsc[idx_mn];
if (s.lthreed) {
rss_half[m][n] = interpolation_weight * rss_boundary[m][n];
zcs_half[m][n] = interpolation_weight * zcs_boundary[m][n];
}
if (s.lasym) {
rsc_half[m][n] = interpolation_weight * rsc_boundary[m][n];
zcc_half[m][n] = interpolation_weight * zcc_boundary[m][n];
if (s.lthreed) {
rcs_half[m][n] = interpolation_weight * rbcs[idx_mn];
zss_half[m][n] = interpolation_weight * zbss[idx_mn];
}
}
}
}
}

for (int k = 0; k < s.nZeta; ++k) {

const int n_theta_to_fill = s.lasym ? s.nThetaEven : s.nThetaReduced;
for (int l = 0; l < n_theta_to_fill; ++l) {
const bool reflected = l >= s.nThetaReduced;
const int l_basis = reflected ? (s.nThetaEven - l) % s.nThetaEven : l;
const double theta_parity = reflected ? -1.0 : 1.0;

w.r_lcfs[k][l] = 0.0;
w.z_lcfs[k][l] = 0.0;
w.d_r_d_theta_lcfs[k][l] = 0.0;
w.d_z_d_theta_lcfs[k][l] = 0.0;

w.r_half[k][l] = 0.0;
w.z_half[k][l] = 0.0;
w.d_r_d_theta_half[k][l] = 0.0;
w.d_z_d_theta_half[k][l] = 0.0;

for (int m = 0; m < s.mpol; ++m) {
for (int n = 0; n <= s.ntor; ++n) {
int idx_ml = m * s.nThetaReduced + l_basis;
int idx_kn = k * (s.nnyq2 + 1) + n;
int idx_mn = m * (s.ntor + 1) + n;

const double basis_norm = 1.0 / (t.mscale[m] * t.nscale[n]);

w.r_lcfs[k][l] +=
basis_norm * rbcc[idx_mn] * t.cosmu[idx_ml] * t.cosnv[idx_kn];
w.z_lcfs[k][l] += basis_norm * zbsc[idx_mn] * theta_parity *
t.sinmu[idx_ml] * t.cosnv[idx_kn];
w.d_r_d_theta_lcfs[k][l] += basis_norm * rbcc[idx_mn] * theta_parity *
t.sinmum[idx_ml] * t.cosnv[idx_kn];
w.d_z_d_theta_lcfs[k][l] +=
basis_norm * zbsc[idx_mn] * t.cosmum[idx_ml] * t.cosnv[idx_kn];
w.r_half[k][l] +=
basis_norm * rcc_half[m][n] * t.cosmu[idx_ml] * t.cosnv[idx_kn];
w.z_half[k][l] += basis_norm * zsc_half[m][n] * theta_parity *
t.sinmu[idx_ml] * t.cosnv[idx_kn];
w.d_r_d_theta_half[k][l] += basis_norm * rcc_half[m][n] *
theta_parity * t.sinmum[idx_ml] *
t.cosnv[idx_kn];
w.d_z_d_theta_half[k][l] +=
basis_norm * zsc_half[m][n] * t.cosmum[idx_ml] * t.cosnv[idx_kn];
if (s.lthreed) {
w.r_lcfs[k][l] += basis_norm * rss_boundary[m][n] * theta_parity *
t.sinmu[idx_ml] * t.sinnv[idx_kn];
w.z_lcfs[k][l] += basis_norm * zcs_boundary[m][n] *
t.cosmu[idx_ml] * t.sinnv[idx_kn];
w.d_r_d_theta_lcfs[k][l] += basis_norm * rss_boundary[m][n] *
t.cosmum[idx_ml] * t.sinnv[idx_kn];
w.d_z_d_theta_lcfs[k][l] += basis_norm * zcs_boundary[m][n] *
theta_parity * t.sinmum[idx_ml] *
t.sinnv[idx_kn];

w.r_half[k][l] += basis_norm * rss_half[m][n] * theta_parity *
t.sinmu[idx_ml] * t.sinnv[idx_kn];
w.z_half[k][l] +=
basis_norm * zcs_half[m][n] * t.cosmu[idx_ml] * t.sinnv[idx_kn];
w.d_r_d_theta_half[k][l] += basis_norm * rss_half[m][n] *
t.cosmum[idx_ml] * t.sinnv[idx_kn];
w.d_z_d_theta_half[k][l] += basis_norm * zcs_half[m][n] *
theta_parity * t.sinmum[idx_ml] *
t.sinnv[idx_kn];
}
if (s.lasym) {
w.r_lcfs[k][l] += basis_norm * rsc_boundary[m][n] * theta_parity *
t.sinmu[idx_ml] * t.cosnv[idx_kn];
w.z_lcfs[k][l] += basis_norm * zcc_boundary[m][n] *
t.cosmu[idx_ml] * t.cosnv[idx_kn];
w.d_r_d_theta_lcfs[k][l] += basis_norm * rsc_boundary[m][n] *
t.cosmum[idx_ml] * t.cosnv[idx_kn];
w.d_z_d_theta_lcfs[k][l] += basis_norm * zcc_boundary[m][n] *
theta_parity * t.sinmum[idx_ml] *
t.cosnv[idx_kn];

w.r_half[k][l] += basis_norm * rsc_half[m][n] * theta_parity *
t.sinmu[idx_ml] * t.cosnv[idx_kn];
w.z_half[k][l] +=
basis_norm * zcc_half[m][n] * t.cosmu[idx_ml] * t.cosnv[idx_kn];
w.d_r_d_theta_half[k][l] += basis_norm * rsc_half[m][n] *
t.cosmum[idx_ml] * t.cosnv[idx_kn];
w.d_z_d_theta_half[k][l] += basis_norm * zcc_half[m][n] *
theta_parity * t.sinmum[idx_ml] *
t.cosnv[idx_kn];
if (s.lthreed) {
w.r_lcfs[k][l] +=
basis_norm * rbcs[idx_mn] * t.cosmu[idx_ml] * t.sinnv[idx_kn];
w.z_lcfs[k][l] += basis_norm * zbss[idx_mn] * theta_parity *
t.sinmu[idx_ml] * t.sinnv[idx_kn];
w.d_r_d_theta_lcfs[k][l] += basis_norm * rbcs[idx_mn] *
theta_parity * t.sinmum[idx_ml] *
t.sinnv[idx_kn];
w.d_z_d_theta_lcfs[k][l] += basis_norm * zbss[idx_mn] *
t.cosmum[idx_ml] * t.sinnv[idx_kn];

w.r_half[k][l] += basis_norm * rcs_half[m][n] * t.cosmu[idx_ml] *
t.sinnv[idx_kn];
w.z_half[k][l] += basis_norm * zss_half[m][n] * theta_parity *
t.sinmu[idx_ml] * t.sinnv[idx_kn];
w.d_r_d_theta_half[k][l] += basis_norm * rcs_half[m][n] *
theta_parity * t.sinmum[idx_ml] *
t.sinnv[idx_kn];
w.d_z_d_theta_half[k][l] += basis_norm * zss_half[m][n] *
t.cosmum[idx_ml] * t.sinnv[idx_kn];
}
}
}
}

w.d_r_d_theta_half[k][l] =
(w.d_r_d_theta_lcfs[k][l] + w.d_r_d_theta_half[k][l]) / 2.0;
w.d_z_d_theta_half[k][l] =
(w.d_z_d_theta_lcfs[k][l] + w.d_z_d_theta_half[k][l]) / 2.0;
}
}

if (!s.lasym) {
for (int k = 0; k < s.nZeta; ++k) {
const int k_reversed = (s.nZeta - k) % s.nZeta;

for (int l = 1; l < s.nThetaReduced - 1; ++l) {
const int l_reversed = (s.nThetaEven - l) % s.nThetaEven;

w.r_lcfs[k_reversed][l_reversed] = w.r_lcfs[k][l];
w.z_lcfs[k_reversed][l_reversed] = -w.z_lcfs[k][l];
w.d_r_d_theta_lcfs[k_reversed][l_reversed] = -w.d_r_d_theta_lcfs[k][l];
w.d_z_d_theta_lcfs[k_reversed][l_reversed] = w.d_z_d_theta_lcfs[k][l];

w.r_half[k_reversed][l_reversed] = w.r_half[k][l];
w.z_half[k_reversed][l_reversed] = -w.z_half[k][l];
w.d_r_d_theta_half[k_reversed][l_reversed] = -w.d_r_d_theta_half[k][l];
w.d_z_d_theta_half[k_reversed][l_reversed] = w.d_z_d_theta_half[k][l];
}
}
}

const int m0 = 0;
for (int k = 0; k < s.nZeta; ++k) {

w.r_axis[k] = 0.0;
w.z_axis[k] = 0.0;

for (int n = 0; n <= s.ntor; ++n) {
int idx_kn = k * (s.nnyq2 + 1) + n;

const double basis_norm = 1.0 / (t.mscale[m0] * t.nscale[n]);

w.r_axis[k] += raxis_c[n] * t.cosnv[idx_kn] * basis_norm;
w.z_axis[k] -= zaxis_s[n] * t.sinnv[idx_kn] * basis_norm;
if (s.lasym) {
w.r_axis[k] -= raxis_s[n] * t.sinnv[idx_kn] * basis_norm;
w.z_axis[k] += zaxis_c[n] * t.cosnv[idx_kn] * basis_norm;
}
}
}

const int number_of_planes_to_search = s.lasym ? s.nZeta : s.nZeta / 2 + 1;
for (int k = 0; k < number_of_planes_to_search; ++k) {

const double min_r =
*std::min_element(w.r_lcfs[k].begin(), w.r_lcfs[k].end());
const double max_r =
*std::max_element(w.r_lcfs[k].begin(), w.r_lcfs[k].end());
const double min_z =
*std::min_element(w.z_lcfs[k].begin(), w.z_lcfs[k].end());
const double max_z =
*std::max_element(w.z_lcfs[k].begin(), w.z_lcfs[k].end());

const double delta_r = (max_r - min_r) / (kNumberOfGridPoints - 1.0);
const double delta_z = (max_z - min_z) / (kNumberOfGridPoints - 1.0);

w.new_r_axis[k] = (max_r + min_r) / 2.0;
w.new_z_axis[k] = (max_z + min_z) / 2.0;

for (int l = 0; l < s.nThetaEven; ++l) {
w.d_r_d_s_half[k][l] =
(w.r_lcfs[k][l] - w.r_half[k][l]) / delta_s + w.r_axis[k];
w.d_z_d_s_half[k][l] =
(w.z_lcfs[k][l] - w.z_half[k][l]) / delta_s + w.z_axis[k];
w.tau0[k][l] = w.d_r_d_theta_half[k][l] * w.d_z_d_s_half[k][l] -
w.d_z_d_theta_half[k][l] * w.d_r_d_s_half[k][l];
}

double min_tau = 0.0;

for (int index_z = 0; index_z < kNumberOfGridPoints; ++index_z) {
double z_grid = min_z + index_z * delta_z;

if (!s.lasym && (k == 0 || k == s.nZeta / 2)) {
z_grid = 0.0;
if (index_z > 0) {
break;
}
}

for (int index_r = 0; index_r < kNumberOfGridPoints; ++index_r) {
double r_grid = min_r + index_r * delta_r;

for (int l = 0; l < s.nThetaEven; ++l) {
w.tau[k][l] = sign_of_jacobian *
(w.tau0[k][l] - w.d_r_d_theta_half[k][l] * z_grid +
w.d_z_d_theta_half[k][l] * r_grid);
}

double min_tau_temp =
*std::min_element(w.tau[k].begin(), w.tau[k].end());

if (min_tau_temp > min_tau) {
min_tau = min_tau_temp;
w.new_r_axis[k] = r_grid;
w.new_z_axis[k] = z_grid;
} else if (min_tau_temp == min_tau) {

if (std::abs(w.new_z_axis[k]) > std::abs(z_grid)) {
w.new_z_axis[k] = z_grid;
}
}
}
}
}

if (!s.lasym) {

for (int k = 1; k <= (s.nZeta - 1) / 2; ++k) {
const int k_reversed = (s.nZeta - k) % s.nZeta;
w.new_r_axis[k_reversed] = w.new_r_axis[k];
w.new_z_axis[k_reversed] = -w.new_z_axis[k];
}
}

const double delta_v = 2.0 / s.nZeta;
for (int k = 0; k < s.nZeta; ++k) {
for (int n = 0; n <= s.ntor; ++n) {

int idx_kn = k * (s.nnyq2 + 1) + n;

w.new_raxis_c[n] +=
delta_v * t.cosnv[idx_kn] * w.new_r_axis[k] / t.nscale[n];
w.new_zaxis_s[n] -=
delta_v * t.sinnv[idx_kn] * w.new_z_axis[k] / t.nscale[n];
if (s.lasym) {
w.new_raxis_s[n] -=
delta_v * t.sinnv[idx_kn] * w.new_r_axis[k] / t.nscale[n];
w.new_zaxis_c[n] +=
delta_v * t.cosnv[idx_kn] * w.new_z_axis[k] / t.nscale[n];
}
}
}

w.new_raxis_c[0] /= 2.0;

if (s.ntor > 0 && s.ntor >= s.nZeta / 2) {
w.new_raxis_c[s.nZeta / 2] /= 2.0;
}
if (s.lasym) {
w.new_zaxis_c[0] /= 2.0;
if (s.ntor > 0 && s.ntor >= s.nZeta / 2) {
w.new_zaxis_c[s.nZeta / 2] /= 2.0;
}
}

return w;
}

}

// source: vmecpp/vmec/fourier_coefficients/fourier_coefficients.cc

// header: vmecpp/vmec/fourier_coefficients/fourier_coefficients.h

#ifndef VMECPP_VMEC_FOURIER_COEFFICIENTS_FOURIER_COEFFICIENTS_H_
#define VMECPP_VMEC_FOURIER_COEFFICIENTS_FOURIER_COEFFICIENTS_H_

#include <Eigen/Dense>
#include <cstdio>
#include <optional>
#include <vector>

// header: vmecpp/vmec/radial_partitioning/radial_partitioning.h

#ifndef VMECPP_VMEC_RADIAL_PARTITIONING_RADIAL_PARTITIONING_H_
#define VMECPP_VMEC_RADIAL_PARTITIONING_RADIAL_PARTITIONING_H_

#include <mutex>
#include <vector>

namespace vmecpp {

class RadialPartitioning {
public:
RadialPartitioning();

void adjustRadialPartitioning(int num_threads, int thread_id, int ns,
bool lfreeb, bool printout = true);

int get_num_threads() const;

int get_thread_id() const;

bool has_boundary() const;

int nsMinF1;

int nsMaxF1;

int nsMinH;

int nsMaxH;

int nsMinFi;

int nsMaxFi;

int nsMinF;

int nsMaxF;

int nsMaxFIncludingLcfs;

private:
int num_threads_;
int thread_id_;

int ns_;
};

}

#endif

namespace vmecpp {

class FourierCoeffs {
public:
FourierCoeffs(const Sizes* s, const RadialPartitioning* r, int nsMin,
int nsMax, int ns);
FourierCoeffs(const FourierCoeffs& other) = default;
FourierCoeffs(FourierCoeffs&& other) noexcept = default;
FourierCoeffs& operator=(const FourierCoeffs& other);
FourierCoeffs& operator=(FourierCoeffs&& other) noexcept;

void setZero();

void decomposeInto(FourierCoeffs& m_x, const Eigen::VectorXd& scalxc) const;

void m1Constraint(double scalingFactor, int sign_of_jacobian,
std::optional<int> jMax = std::nullopt);

void maskGeometryAbove(int mpolGeom, int ntorGeom);

double rzNorm(bool includeOffset, int nsMinHere, int nsMaxHere) const;

double GetXcElement(int rzl, int basis_index, int j, int n, int m) const;

int nsMin() const;
int nsMax() const;

protected:
const Sizes& s_;
const RadialPartitioning& r_;

const int nsMin_;
const int nsMax_;
const int ns;

Eigen::VectorXd rcc;

Eigen::VectorXd rss;

Eigen::VectorXd rsc;

Eigen::VectorXd rcs;

Eigen::VectorXd zsc;

Eigen::VectorXd zcs;

Eigen::VectorXd zcc;

Eigen::VectorXd zss;

Eigen::VectorXd lsc;

Eigen::VectorXd lcs;

Eigen::VectorXd lcc;

Eigen::VectorXd lss;
};

}

#endif

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"

namespace vmecpp {

FourierCoeffs::FourierCoeffs(const Sizes* s, const RadialPartitioning* r,
int nsMin, int nsMax, int ns)
: s_(*s), r_(*r), nsMin_(nsMin), nsMax_(nsMax), ns(ns) {

int jMaxIncludingBoundary = nsMax;
if (r_.nsMaxF1 == ns) {
jMaxIncludingBoundary = ns;
}

int num_fc_RZ = (jMaxIncludingBoundary - nsMin) * s_.mpol * (s_.ntor + 1);
int num_fc_L = (jMaxIncludingBoundary - nsMin) * s_.mpol * (s_.ntor + 1);

rcc.resize(num_fc_RZ);
rcc.setZero();
zsc.resize(num_fc_RZ);
zsc.setZero();
lsc.resize(num_fc_L);
lsc.setZero();
if (s_.lthreed) {
rss.resize(num_fc_RZ);
rss.setZero();
zcs.resize(num_fc_RZ);
zcs.setZero();
lcs.resize(num_fc_L);
lcs.setZero();
}
if (s_.lasym) {
rsc.resize(num_fc_RZ);
rsc.setZero();
zcc.resize(num_fc_RZ);
zcc.setZero();
lcc.resize(num_fc_L);
lcc.setZero();
if (s_.lthreed) {
rcs.resize(num_fc_RZ);
rcs.setZero();
zss.resize(num_fc_RZ);
zss.setZero();
lss.resize(num_fc_L);
lss.setZero();
}
}
}

FourierCoeffs& FourierCoeffs::operator=(const FourierCoeffs& other) {
if (this != &other) {
rcc = other.rcc;
rss = other.rss;
rsc = other.rsc;
rcs = other.rcs;
zsc = other.zsc;
zcs = other.zcs;
zcc = other.zcc;
zss = other.zss;
lsc = other.lsc;
lcs = other.lcs;
lcc = other.lcc;
lss = other.lss;
}
return *this;
}

FourierCoeffs& FourierCoeffs::operator=(FourierCoeffs&& other) noexcept {
if (this != &other) {
rcc = std::move(other.rcc);
rss = std::move(other.rss);
rsc = std::move(other.rsc);
rcs = std::move(other.rcs);
zsc = std::move(other.zsc);
zcs = std::move(other.zcs);
zcc = std::move(other.zcc);
zss = std::move(other.zss);
lsc = std::move(other.lsc);
lcs = std::move(other.lcs);
lcc = std::move(other.lcc);
lss = std::move(other.lss);
}
return *this;
}

int FourierCoeffs::nsMin() const { return nsMin_; }

int FourierCoeffs::nsMax() const { return nsMax_; }

void FourierCoeffs::setZero() {
absl::c_fill(rcc, 0);
absl::c_fill(zsc, 0);
absl::c_fill(lsc, 0);
if (s_.lthreed) {
absl::c_fill(rss, 0);
absl::c_fill(zcs, 0);
absl::c_fill(lcs, 0);
}
if (s_.lasym) {
absl::c_fill(rsc, 0);
absl::c_fill(zcc, 0);
absl::c_fill(lcc, 0);
if (s_.lthreed) {
absl::c_fill(rcs, 0);
absl::c_fill(zss, 0);
absl::c_fill(lss, 0);
}
}
}

void FourierCoeffs::decomposeInto(FourierCoeffs& m_x,
const Eigen::VectorXd& scalxc) const {

int jMaxIncludingBoundary = nsMax_;
if (r_.nsMaxF1 == ns) {
jMaxIncludingBoundary = ns;
}

int jMaxRZ = jMaxIncludingBoundary;

for (int jF = nsMin_; jF < jMaxIncludingBoundary; ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
for (int n = 0; n < s_.ntor + 1; ++n) {
int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

int m_parity = m % 2;

double scal = scalxc[(jF - r_.nsMinF1) * 2 + m_parity];

if (jF < jMaxRZ) {
m_x.rcc[idx_fc] = rcc[idx_fc] * scal;
m_x.zsc[idx_fc] = zsc[idx_fc] * scal;
}
m_x.lsc[idx_fc] = lsc[idx_fc] * scal;
if (s_.lthreed) {
if (jF < jMaxRZ) {
m_x.rss[idx_fc] = rss[idx_fc] * scal;
m_x.zcs[idx_fc] = zcs[idx_fc] * scal;
}
m_x.lcs[idx_fc] = lcs[idx_fc] * scal;
}
if (s_.lasym) {
if (jF < jMaxRZ) {
m_x.rsc[idx_fc] = rsc[idx_fc] * scal;
m_x.zcc[idx_fc] = zcc[idx_fc] * scal;
}
m_x.lcc[idx_fc] = lcc[idx_fc] * scal;
if (s_.lthreed) {
if (jF < jMaxRZ) {
m_x.rcs[idx_fc] = rcs[idx_fc] * scal;
m_x.zss[idx_fc] = zss[idx_fc] * scal;
}
m_x.lss[idx_fc] = lss[idx_fc] * scal;
}
}
}
}
}
}

void FourierCoeffs::m1Constraint(double scalingFactor, int sign_of_jacobian,
std::optional<int> jMax) {
const double sigma = -sign_of_jacobian;
int nsMaxToUse = nsMax_;
if (jMax.has_value()) {
nsMaxToUse = std::min(jMax.value(), nsMaxToUse);
}

for (int jF = nsMin_; jF < nsMaxToUse; ++jF) {
for (int n = 0; n < s_.ntor + 1; ++n) {
int m = 1;
int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;
if (s_.lthreed) {
double old_rss = rss[idx_fc];
rss[idx_fc] = (old_rss + sigma * zcs[idx_fc]) * scalingFactor;
zcs[idx_fc] = (sigma * old_rss - zcs[idx_fc]) * scalingFactor;
}
if (s_.lasym) {
double old_rsc = rsc[idx_fc];
rsc[idx_fc] = (old_rsc + sigma * zcc[idx_fc]) * scalingFactor;
zcc[idx_fc] = (sigma * old_rsc - zcc[idx_fc]) * scalingFactor;
}
}
}
}

void FourierCoeffs::maskGeometryAbove(int mpolGeom, int ntorGeom) {

int jMaxIncludingBoundary = nsMax_;
if (r_.nsMaxF1 == ns) {
jMaxIncludingBoundary = ns;
}

for (int jF = nsMin_; jF < jMaxIncludingBoundary; ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
for (int n = 0; n < s_.ntor + 1; ++n) {
if (m < mpolGeom && n <= ntorGeom) {
continue;
}

int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

rcc[idx_fc] = 0.0;
zsc[idx_fc] = 0.0;
if (s_.lthreed) {
rss[idx_fc] = 0.0;
zcs[idx_fc] = 0.0;
}
if (s_.lasym) {
rsc[idx_fc] = 0.0;
zcc[idx_fc] = 0.0;
if (s_.lthreed) {
rcs[idx_fc] = 0.0;
zss[idx_fc] = 0.0;
}
}

}
}
}
}

double FourierCoeffs::rzNorm(bool include_offset, int nsMinHere,
int nsMaxHere) const {

double local_norm2 = 0.0;

for (int jF = nsMinHere; jF < nsMaxHere; ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
for (int n = 0; n < s_.ntor + 1; ++n) {
int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

if (n > 0 || m > 0 || include_offset) {
local_norm2 += rcc[idx_fc] * rcc[idx_fc];
}
local_norm2 += zsc[idx_fc] * zsc[idx_fc];
if (s_.lthreed) {
local_norm2 += rss[idx_fc] * rss[idx_fc];
local_norm2 += zcs[idx_fc] * zcs[idx_fc];
}
if (s_.lasym) {
local_norm2 += rsc[idx_fc] * rsc[idx_fc];
if (n > 0 || m > 0 || include_offset) {
local_norm2 += zcc[idx_fc] * zcc[idx_fc];
}
if (s_.lthreed) {
local_norm2 += rcs[idx_fc] * rcs[idx_fc];
local_norm2 += zss[idx_fc] * zss[idx_fc];
}
}
}
}
}

return local_norm2;
}

double FourierCoeffs::GetXcElement(int rzl, int idx_basis, int jF, int n,
int m) const {
int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

if (rzl == 0) {
if (idx_basis == 0) {
return rcc[idx_fc];
}
if (s_.lthreed) {
if (idx_basis == 1) {
return rss[idx_fc];
}
}
if (s_.lasym) {
if ((!s_.lthreed && idx_basis == 1) || (s_.lthreed && idx_basis == 2)) {
return rsc[idx_fc];
}
if (s_.lthreed) {
if (idx_basis == 3) {
return rcs[idx_fc];
}
}
}
} else if (rzl == 1) {
if (idx_basis == 0) {
return zsc[idx_fc];
}
if (s_.lthreed) {
if (idx_basis == 1) {
return zcs[idx_fc];
}
}
if (s_.lasym) {
if ((!s_.lthreed && idx_basis == 1) || (s_.lthreed && idx_basis == 2)) {
return zcc[idx_fc];
}
if (s_.lthreed) {
if (idx_basis == 3) {
return zss[idx_fc];
}
}
}
} else if (rzl == 2) {
if (idx_basis == 0) {
return lsc[idx_fc];
}
if (s_.lthreed) {
if (idx_basis == 1) {
return lcs[idx_fc];
}
}
if (s_.lasym) {
if ((!s_.lthreed && idx_basis == 1) || (s_.lthreed && idx_basis == 2)) {
return lcc[idx_fc];
}
if (s_.lthreed) {
if (idx_basis == 3) {
return lss[idx_fc];
}
}
}
}

std::stringstream error_message;
error_message << "did not find";
error_message << " rzl=" << rzl;
error_message << " idx_basis=" << idx_basis;
error_message << " jF=" << jF;
error_message << " n=" << n;
error_message << " m=" << m;
LOG(FATAL) << error_message.str();

__builtin_unreachable();
return 0.0;
}

}

// source: vmecpp/vmec/fourier_forces/fourier_forces.cc

// header: vmecpp/vmec/fourier_forces/fourier_forces.h

#ifndef VMECPP_VMEC_FOURIER_FORCES_FOURIER_FORCES_H_
#define VMECPP_VMEC_FOURIER_FORCES_FOURIER_FORCES_H_

#include <Eigen/Dense>
#include <span>

namespace vmecpp {

class FourierForces : public FourierCoeffs {
public:
FourierForces(const Sizes* s, const RadialPartitioning* r, int ns);
FourierForces(const FourierForces& other);
FourierForces& operator=(const FourierForces& other);
FourierForces(FourierForces&& other) noexcept;
FourierForces& operator=(FourierForces&& other) noexcept;

void zeroZForceForM1();
void residuals(Eigen::Vector3d& fRes, bool includeEdgeRZ) const;

std::span<double> frcc;
std::span<double> frss;
std::span<double> frsc;
std::span<double> frcs;

std::span<double> fzsc;
std::span<double> fzcs;
std::span<double> fzcc;
std::span<double> fzss;

std::span<double> flsc;
std::span<double> flcs;
std::span<double> flcc;
std::span<double> flss;

private:
void BindSpans();
};

}

#endif

#include <algorithm>
#include <utility>
#include <vector>

namespace vmecpp {

FourierForces::FourierForces(const Sizes* s, const RadialPartitioning* r,
int ns)
: FourierCoeffs(s, r, r->nsMinF, r->nsMaxF, ns),
frcc(rcc.data(), rcc.size()),
frss(rss.data(), rss.size()),
frsc(rsc.data(), rsc.size()),
frcs(rcs.data(), rcs.size()),
fzsc(zsc.data(), zsc.size()),
fzcs(zcs.data(), zcs.size()),
fzcc(zcc.data(), zcc.size()),
fzss(zss.data(), zss.size()),
flsc(lsc.data(), lsc.size()),
flcs(lcs.data(), lcs.size()),
flcc(lcc.data(), lcc.size()),
flss(lss.data(), lss.size()) {}

FourierForces::FourierForces(const FourierForces& other)
: FourierCoeffs(other),
frcc(rcc),
frss(rss),
frsc(rsc),
frcs(rcs),
fzsc(zsc),
fzcs(zcs),
fzcc(zcc),
fzss(zss),
flsc(lsc),
flcs(lcs),
flcc(lcc),
flss(lss) {}

void FourierForces::BindSpans() {
frcc = rcc;
frss = rss;
frsc = rsc;
frcs = rcs;
fzsc = zsc;
fzcs = zcs;
fzcc = zcc;
fzss = zss;
flsc = lsc;
flcs = lcs;
flcc = lcc;
flss = lss;
}

FourierForces& FourierForces::operator=(const FourierForces& other) {
if (this != &other) {
FourierCoeffs::operator=(other);
BindSpans();
}
return *this;
}

FourierForces::FourierForces(FourierForces&& other) noexcept
: FourierCoeffs(std::move(other)),
frcc(rcc),
frss(rss),
frsc(rsc),
frcs(rcs),
fzsc(zsc),
fzcs(zcs),
fzcc(zcc),
fzss(zss),
flsc(lsc),
flcs(lcs),
flcc(lcc),
flss(lss) {}

FourierForces& FourierForces::operator=(FourierForces&& other) noexcept {
if (this != &other) {
FourierCoeffs::operator=(std::move(other));
BindSpans();
}
return *this;
}

void FourierForces::zeroZForceForM1() {
for (int jF = nsMin_; jF < nsMax_; ++jF) {
for (int n = 0; n < s_.ntor + 1; ++n) {
int m = 1;
int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;
if (s_.lthreed) {
fzcs[idx_fc] = 0.0;
}
if (s_.lasym) {
fzcc[idx_fc] = 0.0;
}
}
}
}

void FourierForces::residuals(Eigen::Vector3d& fRes,
bool includeEdgeRZForces) const {
int jMaxRZ = std::min(nsMax_, ns - 1);
if (includeEdgeRZForces && r_.nsMaxF1 == ns) {
jMaxRZ = ns;
}

int jMaxIncludeBoundary = nsMax_;
if (r_.nsMaxF1 == ns) {
jMaxIncludeBoundary = ns;
}

double local_fResR = 0.0;
double local_fResZ = 0.0;
double local_fResL = 0.0;
for (int jF = nsMin_; jF < jMaxIncludeBoundary; ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
for (int n = 0; n < s_.ntor + 1; ++n) {
int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

if (jF < jMaxRZ) {
local_fResR += frcc[idx_fc] * frcc[idx_fc];
local_fResZ += fzsc[idx_fc] * fzsc[idx_fc];
}
local_fResL += flsc[idx_fc] * flsc[idx_fc];
if (s_.lthreed) {
if (jF < jMaxRZ) {
local_fResR += frss[idx_fc] * frss[idx_fc];
local_fResZ += fzcs[idx_fc] * fzcs[idx_fc];
}
local_fResL += flcs[idx_fc] * flcs[idx_fc];
}
if (s_.lasym) {
if (jF < jMaxRZ) {
local_fResR += frsc[idx_fc] * frsc[idx_fc];
local_fResZ += fzcc[idx_fc] * fzcc[idx_fc];
}
local_fResL += flcc[idx_fc] * flcc[idx_fc];
if (s_.lthreed) {
if (jF < jMaxRZ) {
local_fResR += frcs[idx_fc] * frcs[idx_fc];
local_fResZ += fzss[idx_fc] * fzss[idx_fc];
}
local_fResL += flss[idx_fc] * flss[idx_fc];
}
}
}
}
}

fRes[0] = local_fResR;
fRes[1] = local_fResZ;
fRes[2] = local_fResL;
}

}

// source: vmecpp/vmec/fourier_geometry/fourier_geometry.cc

// header: vmecpp/vmec/fourier_geometry/fourier_geometry.h

#ifndef VMECPP_VMEC_FOURIER_GEOMETRY_FOURIER_GEOMETRY_H_
#define VMECPP_VMEC_FOURIER_GEOMETRY_FOURIER_GEOMETRY_H_

#include <span>

// header: vmecpp/vmec/radial_profiles/radial_profiles.h

#ifndef VMECPP_VMEC_RADIAL_PROFILES_RADIAL_PROFILES_H_
#define VMECPP_VMEC_RADIAL_PROFILES_RADIAL_PROFILES_H_

#include <Eigen/Dense>
#include <cfloat>
#include <cmath>
#include <string>

#include "absl/status/status.h"

// header: vmecpp/vmec/handover_storage/handover_storage.h

#ifndef VMECPP_VMEC_HANDOVER_STORAGE_HANDOVER_STORAGE_H_
#define VMECPP_VMEC_HANDOVER_STORAGE_HANDOVER_STORAGE_H_

#include <Eigen/Dense>
#include <cstddef>
#include <span>
#include <vector>

#include "absl/status/status.h"

namespace vmecpp {

struct SpectralWidthContribution {
double numerator = 0.0;
double denominator = 0.0;
};

struct RadialExtent {
double r_outer = 0.0;
double r_inner = 0.0;
};

struct GeometricOffset {
double r_00 = 0.0;
double z_00 = 0.0;
};

class HandoverStorage {
public:
explicit HandoverStorage(const Sizes* s);

void allocate(const RadialPartitioning& r, int ns);

void ResetSpectralWidthAccumulators();
void RegisterSpectralWidthContribution(
const SpectralWidthContribution& spectral_width_contribution);

double* SpectralWidthNumerator() { return &spectral_width_numerator_; }
double* SpectralWidthDenominator() { return &spectral_width_denominator_; }
double VolumeAveragedSpectralWidth() const;

void SetRadialExtent(const RadialExtent& radial_extent);
void SetGeometricOffset(const GeometricOffset& geometric_offset);

RadialExtent GetRadialExtent() const;
GeometricOffset GetGeometricOffset() const;

double thermalEnergy;
double magneticEnergy;
double mhdEnergy;

double plasmaVolume;

double voli;

double fNormRZ;

double fNormL;

double fNorm1;

double rBtor0;

double rBtor;

double cTor;

double bSubUVac;

double bSubVVac;

Eigen::VectorXd rCon_LCFS;
Eigen::VectorXd zCon_LCFS;

RowMatrixXd thread_reduce_slots;

RowMatrixXd rmncc_i;
RowMatrixXd rmnss_i;
RowMatrixXd zmnsc_i;
RowMatrixXd zmncs_i;
RowMatrixXd lmnsc_i;
RowMatrixXd lmncs_i;

RowMatrixXd rmnsc_i;
RowMatrixXd rmncs_i;
RowMatrixXd zmncc_i;
RowMatrixXd zmnss_i;
RowMatrixXd lmncc_i;
RowMatrixXd lmnss_i;

RowMatrixXd rmncc_o;
RowMatrixXd rmnss_o;
RowMatrixXd zmnsc_o;
RowMatrixXd zmncs_o;
RowMatrixXd lmnsc_o;
RowMatrixXd lmncs_o;

RowMatrixXd rmnsc_o;
RowMatrixXd rmncs_o;
RowMatrixXd zmncc_o;
RowMatrixXd zmnss_o;
RowMatrixXd lmncc_o;
RowMatrixXd lmnss_o;

int mnsize;
RowMatrixXd all_ar;
RowMatrixXd all_az;
RowMatrixXd all_dr;
RowMatrixXd all_dz;
RowMatrixXd all_br;
RowMatrixXd all_bz;

std::vector<RowMatrixXd> all_cr;
std::vector<RowMatrixXd> all_cz;

RowMatrixXd handover_cR;
Eigen::VectorXd handover_aR;
RowMatrixXd handover_cZ;
Eigen::VectorXd handover_aZ;

Eigen::VectorXd rAxis;
Eigen::VectorXd zAxis;

Eigen::VectorXd rCC_LCFS;
Eigen::VectorXd rSS_LCFS;
Eigen::VectorXd rSC_LCFS;
Eigen::VectorXd rCS_LCFS;
Eigen::VectorXd zSC_LCFS;
Eigen::VectorXd zCS_LCFS;
Eigen::VectorXd zCC_LCFS;
Eigen::VectorXd zSS_LCFS;

Eigen::VectorXd vacuum_magnetic_pressure;

Eigen::VectorXd initial_plasma_pressure_at_boundary;

Eigen::VectorXd initial_vacuum_pressure_at_boundary;

Eigen::VectorXd edge_total_pressure;

Eigen::VectorXd vacuum_b_r;

Eigen::VectorXd vacuum_b_phi;

Eigen::VectorXd vacuum_b_z;

Eigen::VectorXd vacuum_potential;

bool vacuum_reached_checkpoint = false;

absl::Status vacuum_status = absl::OkStatus();

private:
const Sizes& s_;

int num_threads_;
int num_basis_;

double spectral_width_numerator_;
double spectral_width_denominator_;

RadialExtent radial_extent_;
GeometricOffset geometric_offset_;
};

}

#endif

// header: vmecpp/vmec/vmec_constants/vmec_constants.h

#ifndef VMECPP_VMEC_VMEC_CONSTANTS_VMEC_CONSTANTS_H_
#define VMECPP_VMEC_VMEC_CONSTANTS_VMEC_CONSTANTS_H_

namespace vmecpp {

struct VmecConstants {

double rmsPhiP;

double lamscale;

VmecConstants();

void reset();
};

}

#endif

namespace vmecpp {

class RadialProfiles {
public:
RadialProfiles(const RadialPartitioning* s, HandoverStorage* m_h,
const VmecINDATA* id, const FlowControl* fc,
int signOfJacobian, double pDamp);

void setupInputProfiles();

void evalRadialProfiles(bool haveToFlipTheta, VmecConstants& m_vmecconst);

ProfileParameterization findParameterization(const std::string& name,
ProfileType intendedType);
std::string profileTypeToString(ProfileType profileType);

void computeMagneticFluxes();
double torfluxDeriv(double x);
double torflux(double x);
double polfluxDeriv(double x);
double polflux(double x);

double evalMassProfile(double x);
double evalIotaProfile(double x);
double evalCurrProfile(double x);

absl::Status CheckCurrentProfileEnclosesEdgeCurrent();

double evalProfileFunction(const ProfileParameterization& param,
const Eigen::VectorXd& coeffs,
const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues,
bool shouldIntegrate, double normX);

double evalPowerSeries(const Eigen::VectorXd& coeffs, double x,
bool should_integrate);
double evalPowerSeriesI(const Eigen::VectorXd& coeffs, double x);
double evalGaussTrunc(const Eigen::VectorXd& coeffs, double x,
bool shouldIntegrate);
double evalSumAtan(const Eigen::VectorXd& coeffs, double x);
double evalTwoLorentz(const Eigen::VectorXd& coeffs, double x);
double evalTwoPower(const Eigen::VectorXd& coeffs, double x,
bool shouldIntegrate);
double evalTwoPowerGs(const Eigen::VectorXd& coeffs, double x,
bool shouldIntegrate);
double evalAkima(const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues, double x);
double evalAkimaIntegrated(const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues, double x);
double evalCubic(const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues, double x);
double evalCubicIntegrated(const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues, double x);
double evalPedestal(const Eigen::VectorXd& coeffs, double x,
bool shouldIntegrate);
double evalRational(const Eigen::VectorXd& coeffs, double x);
double evalLineSegment(const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues, double x);
double evalLineSegmentIntegrated(const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues,
double x);
double evalNiceQuadratic(const Eigen::VectorXd& coeffs, double x);

void AccumulateVolumeAveragedSpectralWidth() const;

ProfileParameterization pmassType;
ProfileParameterization piotaType;
ProfileParameterization pcurrType;

Eigen::VectorXd phipH;
Eigen::VectorXd chipH;
Eigen::VectorXd iotaH;
Eigen::VectorXd currH;
Eigen::VectorXd massH;
Eigen::VectorXd sqrtSH;

Eigen::VectorXd phipF;
Eigen::VectorXd chipF;
Eigen::VectorXd iotaF;
Eigen::VectorXd sqrtSF;
Eigen::VectorXd radialBlending;

double currv;
double Itor;

double maxToroidalFlux;
double maxPoloidalFlux;

double pressureScalingFactor;

Eigen::VectorXd sm;

Eigen::VectorXd sp;

Eigen::VectorXd dVdsH;

Eigen::VectorXd presH;

Eigen::VectorXd bvcoH;

Eigen::VectorXd bucoH;

Eigen::VectorXd jcuruF;

Eigen::VectorXd jcurvF;

Eigen::VectorXd presgradF;

Eigen::VectorXd dVdsF;

Eigen::VectorXd equiF;

Eigen::VectorXd spectral_width;

Eigen::VectorXd scalxc;

private:
const RadialPartitioning& r_;
HandoverStorage& m_h_;
const VmecINDATA& id_;
const FlowControl& fc_;

const int signOfJacobian;
const double pDamp;

};

}

#endif

namespace vmecpp {

class FourierGeometry : public FourierCoeffs {
public:
FourierGeometry(const Sizes* s, const RadialPartitioning* r, int ns);
FourierGeometry(const FourierGeometry& other);
FourierGeometry& operator=(const FourierGeometry& other);
FourierGeometry(FourierGeometry&& other) noexcept;
FourierGeometry& operator=(FourierGeometry&& other) noexcept;

void interpFromBoundaryAndAxis(const FourierBasisFastPoloidal& t,
const Boundaries& b, const RadialProfiles& p);

void InitFromState(const FourierBasisFastPoloidal& fb,
const RowMatrixXd& rmnc, const RowMatrixXd& zmns,
const RowMatrixXd& lmns_full, const RowMatrixXd& rmns,
const RowMatrixXd& zmnc, const RowMatrixXd& lmnc_full,
const RadialProfiles& p, const VmecConstants& constants,
int sign_of_jacobian, const Boundaries* b = nullptr);

void extrapolateTowardsAxis();

void extrapolateTowardsAxisTranspose();

void ComputeSpectralWidth(const FourierBasisFastPoloidal& fourier_basis,
RadialProfiles& m_radial_profiles, int p = 4,
int q = 1) const;

std::span<double> rmncc;

std::span<double> rmnss;

std::span<double> rmnsc;

std::span<double> rmncs;

std::span<double> zmnsc;

std::span<double> zmncs;

std::span<double> zmncc;

std::span<double> zmnss;

std::span<double> lmnsc;

std::span<double> lmncs;

std::span<double> lmncc;

std::span<double> lmnss;

private:
void BindSpans();
};

}

#endif

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/log/check.h"

namespace vmecpp {

namespace {

std::span<const double> SurfaceOf(std::span<const double> coefficients,
int surface_offset,
int coefficients_per_surface) {
if (coefficients.empty()) {
return {};
}
return coefficients.subspan(surface_offset, coefficients_per_surface);
}

struct ProductBasisSurface {
explicit ProductBasisSurface(int mnsize)
: rcc(mnsize),
rss(mnsize),
rsc(mnsize),
rcs(mnsize),
zsc(mnsize),
zcs(mnsize),
zcc(mnsize),
zss(mnsize) {}

std::vector<double> rcc;
std::vector<double> rss;
std::vector<double> rsc;
std::vector<double> rcs;
std::vector<double> zsc;
std::vector<double> zcs;
std::vector<double> zcc;
std::vector<double> zss;
};
}

FourierGeometry::FourierGeometry(const Sizes* s, const RadialPartitioning* r,
int ns)
: FourierCoeffs(s, r, r->nsMinF1, r->nsMaxF1, ns),
rmncc(rcc.data(), rcc.size()),
rmnss(rss.data(), rss.size()),
rmnsc(rsc.data(), rsc.size()),
rmncs(rcs.data(), rcs.size()),

zmnsc(zsc.data(), zsc.size()),
zmncs(zcs.data(), zcs.size()),
zmncc(zcc.data(), zcc.size()),
zmnss(zss.data(), zss.size()),

lmnsc(lsc.data(), lsc.size()),
lmncs(lcs.data(), lcs.size()),
lmncc(lcc.data(), lcc.size()),
lmnss(lss.data(), lss.size()) {}

FourierGeometry::FourierGeometry(const FourierGeometry& other)
: FourierCoeffs(other),
rmncc(rcc),
rmnss(rss),
rmnsc(rsc),
rmncs(rcs),

zmnsc(zsc),
zmncs(zcs),
zmncc(zcc),
zmnss(zss),

lmnsc(lsc),
lmncs(lcs),
lmncc(lcc),
lmnss(lss) {}

void FourierGeometry::BindSpans() {
rmncc = rcc;
rmnss = rss;
rmnsc = rsc;
rmncs = rcs;
zmnsc = zsc;
zmncs = zcs;
zmncc = zcc;
zmnss = zss;
lmnsc = lsc;
lmncs = lcs;
lmncc = lcc;
lmnss = lss;
}

FourierGeometry& FourierGeometry::operator=(const FourierGeometry& other) {
if (this != &other) {
FourierCoeffs::operator=(other);
BindSpans();
}
return *this;
}

FourierGeometry::FourierGeometry(FourierGeometry&& other) noexcept
: FourierCoeffs(std::move(other)),
rmncc(rcc),
rmnss(rss),
rmnsc(rsc),
rmncs(rcs),

zmnsc(zsc),
zmncs(zcs),
zmncc(zcc),
zmnss(zss),

lmnsc(lsc),
lmncs(lcs),
lmncc(lcc),
lmnss(lss) {}

FourierGeometry& FourierGeometry::operator=(FourierGeometry&& other) noexcept {
if (this != &other) {
FourierCoeffs::operator=(std::move(other));
BindSpans();
}
return *this;
}

void FourierGeometry::interpFromBoundaryAndAxis(
const FourierBasisFastPoloidal& t, const Boundaries& b,
const RadialProfiles& p) {
for (int jF = nsMin_; jF < nsMax_; ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
for (int n = 0; n < s_.ntor + 1; ++n) {
int idx_bdy = m * (s_.ntor + 1) + n;
int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

double basis_norm = 1.0 / (t.mscale[m] * t.nscale[n]);

if (m == 0) {

double interpolationWeight =
p.sqrtSF[jF - r_.nsMinF1] * p.sqrtSF[jF - r_.nsMinF1];

rmncc[idx_fc] =
basis_norm * (interpolationWeight * b.rbcc[idx_bdy] +
(1.0 - interpolationWeight) * b.raxis_c[n]);

zmnsc[idx_fc] = basis_norm * interpolationWeight * b.zbsc[idx_bdy];
if (s_.lthreed) {

rmnss[idx_fc] = basis_norm * interpolationWeight * b.rbss[idx_bdy];
zmncs[idx_fc] =
basis_norm * (interpolationWeight * b.zbcs[idx_bdy] -
(1.0 - interpolationWeight) * b.zaxis_s[n]);
}
if (s_.lasym) {

rmnsc[idx_fc] = basis_norm * interpolationWeight * b.rbsc[idx_bdy];
zmncc[idx_fc] =
basis_norm * (interpolationWeight * b.zbcc[idx_bdy] +
(1.0 - interpolationWeight) * b.zaxis_c[n]);
if (s_.lthreed) {
rmncs[idx_fc] =
basis_norm * (interpolationWeight * b.rbcs[idx_bdy] -
(1.0 - interpolationWeight) * b.raxis_s[n]);

zmnss[idx_fc] =
basis_norm * interpolationWeight * b.zbss[idx_bdy];
}
}
} else {

double interpolationWeight = pow(p.sqrtSF[jF - r_.nsMinF1], m);

rmncc[idx_fc] = basis_norm * interpolationWeight * b.rbcc[idx_bdy];
zmnsc[idx_fc] = basis_norm * interpolationWeight * b.zbsc[idx_bdy];
if (s_.lthreed) {
rmnss[idx_fc] = basis_norm * interpolationWeight * b.rbss[idx_bdy];
zmncs[idx_fc] = basis_norm * interpolationWeight * b.zbcs[idx_bdy];
}
if (s_.lasym) {
rmnsc[idx_fc] = basis_norm * interpolationWeight * b.rbsc[idx_bdy];
zmncc[idx_fc] = basis_norm * interpolationWeight * b.zbcc[idx_bdy];
if (s_.lthreed) {
rmncs[idx_fc] =
basis_norm * interpolationWeight * b.rbcs[idx_bdy];
zmnss[idx_fc] =
basis_norm * interpolationWeight * b.zbss[idx_bdy];
}
}
}
}
}
}
}

void FourierGeometry::InitFromState(
const FourierBasisFastPoloidal& fb, const RowMatrixXd& rmnc,
const RowMatrixXd& zmns, const RowMatrixXd& lmns_full,
const RowMatrixXd& rmns, const RowMatrixXd& zmnc,
const RowMatrixXd& lmnc_full, const RadialProfiles& p,
const VmecConstants& constants, int sign_of_jacobian, const Boundaries* b) {
if (s_.lasym) {

CHECK_EQ(rmns.cols(), rmnc.cols())
<< "InitFromState: lasym is set but rmns is missing";
CHECK_EQ(zmnc.cols(), zmns.cols())
<< "InitFromState: lasym is set but zmnc is missing";
CHECK_EQ(lmnc_full.cols(), lmns_full.cols())
<< "InitFromState: lasym is set but lmnc_full is missing";
}

const int max_ns_to_set_rz_on_from_state = (b == nullptr) ? ns : ns - 1;
const int max_ns_to_set_rz_on_from_state_locally =
std::min(nsMax_, max_ns_to_set_rz_on_from_state);

const int mnsize = s_.mpol * (s_.ntor + 1);
const auto state_surface = [&](int jF) {
ProductBasisSurface surface(mnsize);

const Eigen::VectorXd rmnc_col = rmnc.col(jF);
const std::vector<double> rmnc_col_vector(
rmnc_col.data(), rmnc_col.data() + rmnc_col.size());
fb.cos_to_cc_ss(rmnc_col_vector, surface.rcc, surface.rss, s_.ntor,
s_.mpol);

const Eigen::VectorXd zmns_col = zmns.col(jF);
const std::vector<double> zmns_col_vector(
zmns_col.data(), zmns_col.data() + zmns_col.size());
fb.sin_to_sc_cs(zmns_col_vector, surface.zsc, surface.zcs, s_.ntor,
s_.mpol);

if (s_.lasym) {
const Eigen::VectorXd rmns_col = rmns.col(jF);
const std::vector<double> rmns_col_vector(
rmns_col.data(), rmns_col.data() + rmns_col.size());
fb.sin_to_sc_cs(rmns_col_vector, surface.rsc, surface.rcs, s_.ntor,
s_.mpol);

const Eigen::VectorXd zmnc_col = zmnc.col(jF);
const std::vector<double> zmnc_col_vector(
zmnc_col.data(), zmnc_col.data() + zmnc_col.size());
fb.cos_to_cc_ss(zmnc_col_vector, surface.zcc, surface.zss, s_.ntor,
s_.mpol);
}
return surface;
};

for (int jF = nsMin_; jF < max_ns_to_set_rz_on_from_state_locally; ++jF) {
const ProductBasisSurface surface = state_surface(jF);
for (int m = 0; m < s_.mpol; ++m) {
for (int n = 0; n < s_.ntor + 1; ++n) {
const int idx_mn = m * (s_.ntor + 1) + n;
const int idx_jmn = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;
rmncc[idx_jmn] = surface.rcc[idx_mn];
zmnsc[idx_jmn] = surface.zsc[idx_mn];
if (s_.lthreed) {
rmnss[idx_jmn] = surface.rss[idx_mn];
zmncs[idx_jmn] = surface.zcs[idx_mn];
}
if (s_.lasym) {
rmnsc[idx_jmn] = surface.rsc[idx_mn];
zmncc[idx_jmn] = surface.zcc[idx_mn];
if (s_.lthreed) {
rmncs[idx_jmn] = surface.rcs[idx_mn];
zmnss[idx_jmn] = surface.zss[idx_mn];
}
}
}
}
}

if (b != nullptr) {

Boundaries boundary = *b;
boundary.ensureM1Constrained(1.0);
const ProductBasisSurface lcfs = state_surface(ns - 1);
for (int jF = nsMin_; jF < max_ns_to_set_rz_on_from_state_locally; ++jF) {
const double sqrt_s = p.sqrtSF[jF - r_.nsMinF1];
for (int m = 0; m < s_.mpol; ++m) {
const double weight = (m == 0) ? sqrt_s * sqrt_s : pow(sqrt_s, m);
for (int n = 0; n < s_.ntor + 1; ++n) {
const int idx_mn = m * (s_.ntor + 1) + n;
const int idx_jmn = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;
const double basis_norm = 1.0 / (fb.mscale[m] * fb.nscale[n]);
rmncc[idx_jmn] +=
weight * (basis_norm * boundary.rbcc[idx_mn] - lcfs.rcc[idx_mn]);
zmnsc[idx_jmn] +=
weight * (basis_norm * boundary.zbsc[idx_mn] - lcfs.zsc[idx_mn]);
if (s_.lthreed) {
rmnss[idx_jmn] += weight * (basis_norm * boundary.rbss[idx_mn] -
lcfs.rss[idx_mn]);
zmncs[idx_jmn] += weight * (basis_norm * boundary.zbcs[idx_mn] -
lcfs.zcs[idx_mn]);
}
if (s_.lasym) {
rmnsc[idx_jmn] += weight * (basis_norm * boundary.rbsc[idx_mn] -
lcfs.rsc[idx_mn]);
zmncc[idx_jmn] += weight * (basis_norm * boundary.zbcc[idx_mn] -
lcfs.zcc[idx_mn]);
if (s_.lthreed) {
rmncs[idx_jmn] += weight * (basis_norm * boundary.rbcs[idx_mn] -
lcfs.rcs[idx_mn]);
zmnss[idx_jmn] += weight * (basis_norm * boundary.zbss[idx_mn] -
lcfs.zss[idx_mn]);
}
}
}
}
}
}

for (int jF = nsMin_; jF < nsMax_; ++jF) {
const Eigen::VectorXd lmns_col = lmns_full.col(jF);
const std::vector<double> lmns_col_vector(
lmns_col.data(), lmns_col.data() + lmns_col.size());
std::vector<double> lmnsc_at_jF(s_.mpol * (s_.ntor + 1));
std::vector<double> lmncs_at_jF(s_.mpol * (s_.ntor + 1));
fb.sin_to_sc_cs(lmns_col_vector, lmnsc_at_jF, lmncs_at_jF, s_.ntor,
s_.mpol);

std::vector<double> lmncc_at_jF(s_.mpol * (s_.ntor + 1));
std::vector<double> lmnss_at_jF(s_.mpol * (s_.ntor + 1));
if (s_.lasym) {
const Eigen::VectorXd lmnc_col = lmnc_full.col(jF);
const std::vector<double> lmnc_col_vector(
lmnc_col.data(), lmnc_col.data() + lmnc_col.size());
fb.cos_to_cc_ss(lmnc_col_vector, lmncc_at_jF, lmnss_at_jF, s_.ntor,
s_.mpol);
}

for (int m = 0; m < s_.mpol; ++m) {
for (int n = 0; n < s_.ntor + 1; ++n) {
const int idx_mn = m * (s_.ntor + 1) + n;
const int idx_jmn = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

const double lambda_unscaling =
constants.lamscale / p.phipF[jF - r_.nsMinF1];

lmnsc[idx_jmn] = lmnsc_at_jF[idx_mn] / lambda_unscaling;
if (s_.lthreed) {
lmncs[idx_jmn] = lmncs_at_jF[idx_mn] / lambda_unscaling;
}
if (s_.lasym) {
lmncc[idx_jmn] = lmncc_at_jF[idx_mn] / lambda_unscaling;
if (s_.lthreed) {
lmnss[idx_jmn] = lmnss_at_jF[idx_mn] / lambda_unscaling;
}
}
}
}
}

if (b != nullptr && r_.has_boundary()) {
const int jF = ns - 1;
const int mnsize = s_.mpol * (s_.ntor + 1);

auto rmncc_begin = rmncc.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
std::copy(b->rbcc.begin(), b->rbcc.begin() + mnsize, rmncc_begin);
auto zmnsc_begin = zmnsc.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
std::copy(b->zbsc.begin(), b->zbsc.begin() + mnsize, zmnsc_begin);

if (s_.lthreed) {
auto rmnss_begin =
rmnss.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
std::copy(b->rbss.begin(), b->rbss.begin() + mnsize, rmnss_begin);

auto zmncs_begin =
zmncs.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
std::copy(b->zbcs.begin(), b->zbcs.begin() + mnsize, zmncs_begin);
}

if (s_.lasym) {
auto rmnsc_begin =
rmnsc.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
std::copy(b->rbsc.begin(), b->rbsc.begin() + mnsize, rmnsc_begin);

auto zmncc_begin =
zmncc.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
std::copy(b->zbcc.begin(), b->zbcc.begin() + mnsize, zmncc_begin);

if (s_.lthreed) {
auto rmncs_begin =
rmncs.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
std::copy(b->rbcs.begin(), b->rbcs.begin() + mnsize, rmncs_begin);

auto zmnss_begin =
zmnss.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
std::copy(b->zbss.begin(), b->zbss.begin() + mnsize, zmnss_begin);
}
}

for (int m = 0; m < s_.mpol; ++m) {
for (int n = 0; n < s_.ntor + 1; ++n) {
int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

double basis_norm = 1.0 / (fb.mscale[m] * fb.nscale[n]);

rmncc[idx_fc] *= basis_norm;
zmnsc[idx_fc] *= basis_norm;

if (s_.lthreed) {
rmnss[idx_fc] *= basis_norm;
zmncs[idx_fc] *= basis_norm;
}
if (s_.lasym) {
rmnsc[idx_fc] *= basis_norm;
zmncc[idx_fc] *= basis_norm;
if (s_.lthreed) {
rmncs[idx_fc] *= basis_norm;
zmnss[idx_fc] *= basis_norm;
}
}
}
}
}

this->m1Constraint(0.5, sign_of_jacobian, max_ns_to_set_rz_on_from_state);

if (nsMin_ == 0) {

const int jF = 0;
const int m = 0;
for (int n = 0; n < s_.ntor + 1; ++n) {
int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;
lmnsc[idx_fc] = 0.0;
if (s_.lthreed) {
lmncs[idx_fc] = 0.0;
}
if (s_.lasym) {
lmncc[idx_fc] = 0.0;
if (s_.lthreed) {
lmnss[idx_fc] = 0.0;
}
}
}
}
}

void FourierGeometry::extrapolateTowardsAxis() {
if (nsMin_ > 0) {

return;
}

int axis = 0;
int firstSurface = 1;
for (int n = 0; n < s_.ntor + 1; ++n) {
int m0 = 0;
int m1 = 1;

int axis0 = (axis * s_.mpol + m0) * (s_.ntor + 1) + n;
int axis1 = (axis * s_.mpol + m1) * (s_.ntor + 1) + n;
int firstSurface0 = (firstSurface * s_.mpol + m0) * (s_.ntor + 1) + n;
int firstSurface1 = (firstSurface * s_.mpol + m1) * (s_.ntor + 1) + n;

rmncc[axis1] = rmncc[firstSurface1];
zmnsc[axis1] = zmnsc[firstSurface1];
lmnsc[axis1] = lmnsc[firstSurface1];
if (s_.lthreed) {
rmnss[axis1] = rmnss[firstSurface1];
zmncs[axis1] = zmncs[firstSurface1];
lmncs[axis1] = lmncs[firstSurface1];

lmncs[axis0] = lmncs[firstSurface0];
}
if (s_.lasym) {
rmnsc[axis1] = rmnsc[firstSurface1];
zmncc[axis1] = zmncc[firstSurface1];
lmncc[axis1] = lmncc[firstSurface1];

lmncc[axis0] = lmncc[firstSurface0];
if (s_.lthreed) {
rmncs[axis1] = rmncs[firstSurface1];
zmnss[axis1] = zmnss[firstSurface1];
lmnss[axis1] = lmnss[firstSurface1];
}
}
}
}

void FourierGeometry::extrapolateTowardsAxisTranspose() {
if (nsMin_ > 0) {
return;
}
int axis = 0;
int firstSurface = 1;
for (int n = 0; n < s_.ntor + 1; ++n) {
int m0 = 0;
int m1 = 1;
int axis0 = (axis * s_.mpol + m0) * (s_.ntor + 1) + n;
int axis1 = (axis * s_.mpol + m1) * (s_.ntor + 1) + n;
int firstSurface0 = (firstSurface * s_.mpol + m0) * (s_.ntor + 1) + n;
int firstSurface1 = (firstSurface * s_.mpol + m1) * (s_.ntor + 1) + n;

auto fold = [](std::span<double> c, int axisIdx, int firstIdx) {
c[firstIdx] += c[axisIdx];
c[axisIdx] = 0.0;
};
fold(rmncc, axis1, firstSurface1);
fold(zmnsc, axis1, firstSurface1);
fold(lmnsc, axis1, firstSurface1);
if (s_.lthreed) {
fold(rmnss, axis1, firstSurface1);
fold(zmncs, axis1, firstSurface1);
fold(lmncs, axis1, firstSurface1);
fold(lmncs, axis0, firstSurface0);
}
if (s_.lasym) {
fold(rmnsc, axis1, firstSurface1);
fold(zmncc, axis1, firstSurface1);
fold(lmncc, axis1, firstSurface1);
fold(lmncc, axis0, firstSurface0);
if (s_.lthreed) {
fold(rmncs, axis1, firstSurface1);
fold(zmnss, axis1, firstSurface1);
fold(lmnss, axis1, firstSurface1);
}
}
}
}

void FourierGeometry::ComputeSpectralWidth(
const FourierBasisFastPoloidal& fourier_basis,
RadialProfiles& m_radial_profiles, const int p, const int q) const {
int minimum_j = nsMin_;
if (nsMin_ == 0) {
minimum_j = 1;

m_radial_profiles.spectral_width[nsMin_ - r_.nsMinF1] = 1.0;
}

const int coefficients_per_surface = s_.mpol * (s_.ntor + 1);

const std::span<const double> mscale(fourier_basis.mscale.data(),
fourier_basis.mscale.size());
const std::span<const double> nscale(fourier_basis.nscale.data(),
fourier_basis.nscale.size());

for (int jF = minimum_j; jF < nsMax_; ++jF) {
const int surface_offset = (jF - nsMin_) * coefficients_per_surface;

const SurfaceFourierGeometry surface = {
.rmncc = SurfaceOf(rmncc, surface_offset, coefficients_per_surface),
.rmnss = SurfaceOf(rmnss, surface_offset, coefficients_per_surface),
.rmnsc = SurfaceOf(rmnsc, surface_offset, coefficients_per_surface),
.rmncs = SurfaceOf(rmncs, surface_offset, coefficients_per_surface),
.zmnsc = SurfaceOf(zmnsc, surface_offset, coefficients_per_surface),
.zmncs = SurfaceOf(zmncs, surface_offset, coefficients_per_surface),
.zmncc = SurfaceOf(zmncc, surface_offset, coefficients_per_surface),
.zmnss = SurfaceOf(zmnss, surface_offset, coefficients_per_surface)};

m_radial_profiles.spectral_width[jF - r_.nsMinF1] =
SpectralWidth(surface, s_, mscale, nscale, p, q);
}
}

}

// source: vmecpp/vmec/fourier_velocity/fourier_velocity.cc

// header: vmecpp/vmec/fourier_velocity/fourier_velocity.h

#ifndef VMECPP_VMEC_FOURIER_VELOCITY_FOURIER_VELOCITY_H_
#define VMECPP_VMEC_FOURIER_VELOCITY_FOURIER_VELOCITY_H_

#include <span>

namespace vmecpp {

class FourierVelocity : public FourierCoeffs {
public:
FourierVelocity(const Sizes* s, const RadialPartitioning* r, int ns);
FourierVelocity(const FourierVelocity& other);
FourierVelocity& operator=(const FourierVelocity& other);
FourierVelocity(FourierVelocity&& other) noexcept;
FourierVelocity& operator=(FourierVelocity&& other) noexcept;

std::span<double> vrcc;
std::span<double> vrss;
std::span<double> vrsc;
std::span<double> vrcs;

std::span<double> vzsc;
std::span<double> vzcs;
std::span<double> vzcc;
std::span<double> vzss;

std::span<double> vlsc;
std::span<double> vlcs;
std::span<double> vlcc;
std::span<double> vlss;

private:
void BindSpans();
};

}

#endif

#include <utility>

namespace vmecpp {

FourierVelocity::FourierVelocity(const Sizes* s, const RadialPartitioning* r,
int ns)
: FourierCoeffs(s, r, r->nsMinF, r->nsMaxF, ns),
vrcc(rcc.data(), rcc.size()),
vrss(rss.data(), rss.size()),
vrsc(rsc.data(), rsc.size()),
vrcs(rcs.data(), rcs.size()),
vzsc(zsc.data(), zsc.size()),
vzcs(zcs.data(), zcs.size()),
vzcc(zcc.data(), zcc.size()),
vzss(zss.data(), zss.size()),
vlsc(lsc.data(), lsc.size()),
vlcs(lcs.data(), lcs.size()),
vlcc(lcc.data(), lcc.size()),
vlss(lss.data(), lss.size()) {}

FourierVelocity::FourierVelocity(const FourierVelocity& other)
: FourierCoeffs(other),
vrcc(rcc),
vrss(rss),
vrsc(rsc),
vrcs(rcs),
vzsc(zsc),
vzcs(zcs),
vzcc(zcc),
vzss(zss),
vlsc(lsc),
vlcs(lcs),
vlcc(lcc),
vlss(lss) {}

void FourierVelocity::BindSpans() {
vrcc = rcc;
vrss = rss;
vrsc = rsc;
vrcs = rcs;
vzsc = zsc;
vzcs = zcs;
vzcc = zcc;
vzss = zss;
vlsc = lsc;
vlcs = lcs;
vlcc = lcc;
vlss = lss;
}

FourierVelocity& FourierVelocity::operator=(const FourierVelocity& other) {
if (this != &other) {
FourierCoeffs::operator=(other);
BindSpans();
}
return *this;
}

FourierVelocity::FourierVelocity(FourierVelocity&& other) noexcept
: FourierCoeffs(std::move(other)),
vrcc(rcc),
vrss(rss),
vrsc(rsc),
vrcs(rcs),
vzsc(zsc),
vzcs(zcs),
vzcc(zcc),
vzss(zss),
vlsc(lsc),
vlcs(lcs),
vlcc(lcc),
vlss(lss) {}

FourierVelocity& FourierVelocity::operator=(FourierVelocity&& other) noexcept {
if (this != &other) {
FourierCoeffs::operator=(std::move(other));
BindSpans();
}
return *this;
}

}

// source: vmecpp/vmec/geometry/geometry.cc

// header: vmecpp/vmec/geometry/geometry.h

#ifndef VMECPP_VMEC_GEOMETRY_GEOMETRY_H_
#define VMECPP_VMEC_GEOMETRY_GEOMETRY_H_

#include <array>
#include <vector>

namespace vmecpp {

struct GeometryDimensions {
int ns;
int mpol;
int ntor;
int nfp;
};

struct GeometryCoefficients {
std::vector<double> r_cc;
std::vector<double> r_ss;
std::vector<double> r_sc;
std::vector<double> r_cs;
std::vector<double> z_sc;
std::vector<double> z_cs;
std::vector<double> z_cc;
std::vector<double> z_ss;
std::vector<double> lambda_sc;
std::vector<double> lambda_cs;
std::vector<double> lambda_cc;
std::vector<double> lambda_ss;
};

struct Geometry {
GeometryDimensions dimensions;
std::vector<double> toroidal_flux;
std::vector<double> poloidal_flux;
GeometryCoefficients coefficients;
};

inline constexpr int kGeometryJetSize = 10;
using GeometryJet = std::array<double, kGeometryJetSize>;

struct GeometryPoint {
GeometryJet r;
GeometryJet z;
GeometryJet lambda;
GeometryJet toroidal_flux;
GeometryJet poloidal_flux;
};

GeometryPoint EvaluateGeometry(const Geometry& geometry, double s, double theta,
double zeta);

}

#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace vmecpp {
namespace {

struct RadialWeights {
std::array<int, 4> indices;
std::array<double, 4> value;
std::array<double, 4> first;
std::array<double, 4> second;
int count;
};

RadialWeights GetRadialWeights(int ns, double s) {
if (ns < 2) {
throw std::invalid_argument("geometry requires at least two radial points");
}
if (s < 0.0 || s > 1.0) {
throw std::out_of_range("s must be in [0, 1]");
}
const double scale = ns - 1;
const double scaled = s * scale;
if (ns < 4) {
const int inner = std::min(static_cast<int>(scaled), ns - 2);
const double outer_weight = scaled - inner;
return {.indices = {inner, inner + 1, 0, 0},
.value = {1.0 - outer_weight, outer_weight, 0.0, 0.0},
.first = {-scale, scale, 0.0, 0.0},
.second = {0.0, 0.0, 0.0, 0.0},
.count = 2};
}
const int start = std::clamp(static_cast<int>(scaled) - 1, 0, ns - 4);
const double x = scaled - start;
RadialWeights result{.indices = {start, start + 1, start + 2, start + 3},
.value = {},
.first = {},
.second = {},
.count = 4};
for (int i = 0; i < 4; ++i) {
double denominator = 1.0;
for (int j = 0; j < 4; ++j) {
if (j != i) denominator *= i - j;
}
double value = 1.0;
for (int j = 0; j < 4; ++j) {
if (j != i) value *= x - j;
}
double first = 0.0;
double second = 0.0;
for (int omitted = 0; omitted < 4; ++omitted) {
if (omitted == i) continue;
double term = 1.0;
for (int j = 0; j < 4; ++j) {
if (j != i && j != omitted) term *= x - j;
}
first += term;
for (int omitted_second = 0; omitted_second < 4; ++omitted_second) {
if (omitted_second == i || omitted_second == omitted) continue;
double second_term = 1.0;
for (int j = 0; j < 4; ++j) {
if (j != i && j != omitted && j != omitted_second) {
second_term *= x - j;
}
}
second += second_term;
}
}
result.value[i] = value / denominator;
result.first[i] = scale * first / denominator;
result.second[i] = scale * scale * second / denominator;
}
return result;
}

int CoefficientIndex(const GeometryDimensions& dimensions, int radial_index,
int m, int n) {
return (radial_index * dimensions.mpol + m) * (dimensions.ntor + 1) + n;
}

void CheckSize(const std::vector<double>& values, int expected,
const char* name, bool may_be_empty = false) {
if ((may_be_empty && values.empty()) ||
values.size() == static_cast<std::size_t>(expected)) {
return;
}
throw std::invalid_argument(std::string(name) + " has wrong size");
}

void Validate(const Geometry& geometry) {
const GeometryDimensions& d = geometry.dimensions;
if (d.ns < 2 || d.mpol < 1 || d.ntor < 0 || d.nfp < 1) {
throw std::invalid_argument("invalid geometry dimensions");
}
CheckSize(geometry.toroidal_flux, d.ns, "toroidal_flux");
CheckSize(geometry.poloidal_flux, d.ns, "poloidal_flux");
const int size = d.ns * d.mpol * (d.ntor + 1);
CheckSize(geometry.coefficients.r_cc, size, "r_cc");
CheckSize(geometry.coefficients.z_sc, size, "z_sc");
CheckSize(geometry.coefficients.lambda_sc, size, "lambda_sc");
CheckSize(geometry.coefficients.r_ss, size, "r_ss", true);
CheckSize(geometry.coefficients.r_sc, size, "r_sc", true);
CheckSize(geometry.coefficients.r_cs, size, "r_cs", true);
CheckSize(geometry.coefficients.z_cs, size, "z_cs", true);
CheckSize(geometry.coefficients.z_cc, size, "z_cc", true);
CheckSize(geometry.coefficients.z_ss, size, "z_ss", true);
CheckSize(geometry.coefficients.lambda_cs, size, "lambda_cs", true);
CheckSize(geometry.coefficients.lambda_cc, size, "lambda_cc", true);
CheckSize(geometry.coefficients.lambda_ss, size, "lambda_ss", true);
}

std::array<double, 3> TrigDerivatives(bool sine, int mode, double angle) {
const double argument = mode * angle;
if (sine) {
return {std::sin(argument), mode * std::cos(argument),
-mode * mode * std::sin(argument)};
}
return {std::cos(argument), -mode * std::sin(argument),
-mode * mode * std::cos(argument)};
}

GeometryJet EvaluateProfile(const std::vector<double>& profile,
const RadialWeights& radial) {
GeometryJet result{};
for (int i = 0; i < radial.count; ++i) {
result[0] += radial.value[i] * profile[radial.indices[i]];
result[1] += radial.first[i] * profile[radial.indices[i]];
result[4] += radial.second[i] * profile[radial.indices[i]];
}
return result;
}

struct BasisJet {
double value;
double ds;
double dtheta;
double dzeta;
double ds_dtheta;
double ds_dzeta;
double dtheta2;
double dtheta_dzeta;
double dzeta2;
double ds2;
};

BasisJet MakeBasisJet(const std::array<double, 3>& poloidal,
const std::array<double, 3>& toroidal,
const std::array<double, 4>& coefficients,
const RadialWeights& radial) {
double coefficient = 0.0;
double coefficient_s = 0.0;
double coefficient_ss = 0.0;
for (int i = 0; i < radial.count; ++i) {
coefficient += radial.value[i] * coefficients[i];
coefficient_s += radial.first[i] * coefficients[i];
coefficient_ss += radial.second[i] * coefficients[i];
}
return {.value = coefficient * poloidal[0] * toroidal[0],
.ds = coefficient_s * poloidal[0] * toroidal[0],
.dtheta = coefficient * poloidal[1] * toroidal[0],
.dzeta = coefficient * poloidal[0] * toroidal[1],
.ds_dtheta = coefficient_s * poloidal[1] * toroidal[0],
.ds_dzeta = coefficient_s * poloidal[0] * toroidal[1],
.dtheta2 = coefficient * poloidal[2] * toroidal[0],
.dtheta_dzeta = coefficient * poloidal[1] * toroidal[1],
.dzeta2 = coefficient * poloidal[0] * toroidal[2],
.ds2 = coefficient_ss * poloidal[0] * toroidal[0]};
}

void AddBasis(const std::vector<double>& coefficients, bool sine_m, bool sine_n,
int m, int n, double theta, double zeta,
const GeometryDimensions& dimensions, const RadialWeights& radial,
GeometryJet& m_value) {
if (coefficients.empty()) return;
const auto poloidal = TrigDerivatives(sine_m, m, theta);
const auto toroidal = TrigDerivatives(sine_n, n * dimensions.nfp, zeta);
std::array<double, 4> radial_coefficients{};
for (int i = 0; i < radial.count; ++i) {
radial_coefficients[i] =
coefficients[CoefficientIndex(dimensions, radial.indices[i], m, n)];
}
const BasisJet basis =
MakeBasisJet(poloidal, toroidal, radial_coefficients, radial);
m_value[0] += basis.value;
m_value[1] += basis.ds;
m_value[2] += basis.dtheta;
m_value[3] += basis.dzeta;
m_value[4] += basis.ds2;
m_value[5] += basis.ds_dtheta;
m_value[6] += basis.ds_dzeta;
m_value[7] += basis.dtheta2;
m_value[8] += basis.dtheta_dzeta;
m_value[9] += basis.dzeta2;
}

void AddQuantity(const GeometryCoefficients& coefficients, char quantity, int m,
int n, double theta, double zeta,
const GeometryDimensions& dimensions,
const RadialWeights& radial, GeometryJet& m_value) {
const std::vector<double>* cc;
const std::vector<double>* ss;
const std::vector<double>* sc;
const std::vector<double>* cs;
if (quantity == 'r') {
cc = &coefficients.r_cc;
ss = &coefficients.r_ss;
sc = &coefficients.r_sc;
cs = &coefficients.r_cs;
} else if (quantity == 'z') {
cc = &coefficients.z_cc;
ss = &coefficients.z_ss;
sc = &coefficients.z_sc;
cs = &coefficients.z_cs;
} else {
cc = &coefficients.lambda_cc;
ss = &coefficients.lambda_ss;
sc = &coefficients.lambda_sc;
cs = &coefficients.lambda_cs;
}
AddBasis(*cc, false, false, m, n, theta, zeta, dimensions, radial, m_value);
AddBasis(*ss, true, true, m, n, theta, zeta, dimensions, radial, m_value);
AddBasis(*sc, true, false, m, n, theta, zeta, dimensions, radial, m_value);
AddBasis(*cs, false, true, m, n, theta, zeta, dimensions, radial, m_value);
}

}

GeometryPoint EvaluateGeometry(const Geometry& geometry, double s, double theta,
double zeta) {
Validate(geometry);
const RadialWeights radial = GetRadialWeights(geometry.dimensions.ns, s);
GeometryPoint result{};
result.toroidal_flux = EvaluateProfile(geometry.toroidal_flux, radial);
result.poloidal_flux = EvaluateProfile(geometry.poloidal_flux, radial);
for (int m = 0; m < geometry.dimensions.mpol; ++m) {
for (int n = 0; n <= geometry.dimensions.ntor; ++n) {
AddQuantity(geometry.coefficients, 'r', m, n, theta, zeta,
geometry.dimensions, radial, result.r);
AddQuantity(geometry.coefficients, 'z', m, n, theta, zeta,
geometry.dimensions, radial, result.z);
AddQuantity(geometry.coefficients, 'l', m, n, theta, zeta,
geometry.dimensions, radial, result.lambda);
}
}
return result;
}

}

// source: vmecpp/vmec/geometry/vmec_geometry.cc

// header: vmecpp/vmec/geometry/vmec_geometry.h

#ifndef VMECPP_VMEC_GEOMETRY_VMEC_GEOMETRY_H_
#define VMECPP_VMEC_GEOMETRY_VMEC_GEOMETRY_H_

#include <cstdint>

// header: vmecpp/vmec/output_quantities/output_quantities.h

#ifndef VMECPP_VMEC_OUTPUT_QUANTITIES_OUTPUT_QUANTITIES_H_
#define VMECPP_VMEC_OUTPUT_QUANTITIES_OUTPUT_QUANTITIES_H_

#include <Eigen/Dense>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

// header: vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_

#include <Eigen/Dense>
#include <climits>
#include <memory>
#include <span>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "absl/status/statusor.h"

// header: vmecpp/vmec/ideal_mhd_model/dft_toroidal.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_DFT_TOROIDAL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_DFT_TOROIDAL_H_

#include <Eigen/Dense>

// header: vmecpp/vmec/ideal_mhd_model/dft_data.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_DFT_DATA_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_DFT_DATA_H_

#include <span>

namespace vmecpp {

struct RealSpaceForces {
std::span<const double> armn_e;
std::span<const double> armn_o;
std::span<const double> azmn_e;
std::span<const double> azmn_o;
std::span<const double> blmn_e;
std::span<const double> blmn_o;
std::span<const double> brmn_e;
std::span<const double> brmn_o;
std::span<const double> bzmn_e;
std::span<const double> bzmn_o;
std::span<const double> clmn_e;
std::span<const double> clmn_o;
std::span<const double> crmn_e;
std::span<const double> crmn_o;
std::span<const double> czmn_e;
std::span<const double> czmn_o;
std::span<const double> frcon_e;
std::span<const double> frcon_o;
std::span<const double> fzcon_e;
std::span<const double> fzcon_o;
};

struct RealSpaceGeometry {
std::span<double> r1_e;
std::span<double> r1_o;
std::span<double> ru_e;
std::span<double> ru_o;
std::span<double> rv_e;
std::span<double> rv_o;
std::span<double> z1_e;
std::span<double> z1_o;
std::span<double> zu_e;
std::span<double> zu_o;
std::span<double> zv_e;
std::span<double> zv_o;
std::span<double> lu_e;
std::span<double> lu_o;
std::span<double> lv_e;
std::span<double> lv_o;
std::span<double> rCon;
std::span<double> zCon;
};

}

#endif

namespace vmecpp {

void FourierToReal3DSymmFastPoloidal(const FourierGeometry& physical_x,
const Eigen::VectorXd& xmpq,
const RadialPartitioning& r,
const Sizes& s, const RadialProfiles& rp,
const FourierBasisFastPoloidal& fb,
RealSpaceGeometry& m_geometry);

void ForcesToFourier3DSymmFastPoloidal(
const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
const FourierBasisFastPoloidal& fb,
VacuumPressureState vacuum_pressure_state,
FourierForces& m_physical_forces);

void FourierToReal3DAsymFastPoloidal(const FourierGeometry& physical_x,
const Eigen::VectorXd& xmpq,
const RadialPartitioning& r,
const Sizes& s, const RadialProfiles& rp,
const FourierBasisFastPoloidal& fb,
RealSpaceGeometry& m_geometry);

void ForcesToFourier3DAsymFastPoloidal(
const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
const FourierBasisFastPoloidal& fb,
VacuumPressureState vacuum_pressure_state,
FourierForces& m_physical_forces);

}

#endif

// header: vmecpp/vmec/ideal_mhd_model/local_force_composition.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_LOCAL_FORCE_COMPOSITION_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_LOCAL_FORCE_COMPOSITION_H_

#include <algorithm>
#include <cmath>

// header: vmecpp/vmec/ideal_mhd_model/bco_kernel.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_BCO_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_BCO_KERNEL_H_

namespace vmecpp {

inline void ComputeBCo(const double* guu, const double* guv, const double* gvv,
const double* bsupu, const double* bsupv, bool lthreed,
int n, double* bsubu, double* bsubv) {
if (lthreed) {
for (int i = 0; i < n; ++i) {
bsubu[i] = guu[i] * bsupu[i] + guv[i] * bsupv[i];
bsubv[i] = guv[i] * bsupu[i] + gvv[i] * bsupv[i];
}
} else {
for (int i = 0; i < n; ++i) {
bsubu[i] = guu[i] * bsupu[i];
bsubv[i] = gvv[i] * bsupv[i];
}
}
}

}

#endif

// header: vmecpp/vmec/ideal_mhd_model/bcontra_kernel.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_BCONTRA_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_BCONTRA_KERNEL_H_

namespace vmecpp {

inline void ComputeBsupContra(const double* lue, const double* luo,
const double* lve, const double* lvo,
const double* gsqrt, const double* sqrtSH,
bool lthreed, int nZnT, int nsMinF1, int nsMinH,
int nsMaxH, double* bsupu, double* bsupv) {
for (int jH = nsMinH; jH < nsMaxH; ++jH) {
const double sH = sqrtSH[jH - nsMinH];
for (int kl = 0; kl < nZnT; ++kl) {
const int i_in = (jH - nsMinF1) * nZnT + kl;
const int i_out = (jH + 1 - nsMinF1) * nZnT + kl;
const int ih = (jH - nsMinH) * nZnT + kl;

if (lthreed) {
bsupu[ih] = 0.5 *
((lve[i_in] + lve[i_out]) + sH * (lvo[i_in] + lvo[i_out])) /
gsqrt[ih];
} else {
bsupu[ih] = 0.0;
}
bsupv[ih] = 0.5 *
((lue[i_in] + lue[i_out]) + sH * (luo[i_in] + luo[i_out])) /
gsqrt[ih];
}
}
}

}

#endif

// header: vmecpp/vmec/ideal_mhd_model/constraint_force_kernel.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_CONSTRAINT_FORCE_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_CONSTRAINT_FORCE_KERNEL_H_

#include <algorithm>

namespace vmecpp {

inline void ComputeEffectiveConstraintForce(
const double* rCon, const double* rCon0, const double* zCon,
const double* zCon0, const double* ruFull, const double* zuFull, int nZnT,
int nsMinF, int nsMaxFIncludingLcfs, double* gConEff) {
int jMin = 0;
if (nsMinF == 0) {
jMin = 1;
}
for (int jF = std::max(jMin, nsMinF); jF < nsMaxFIncludingLcfs; ++jF) {
for (int kl = 0; kl < nZnT; ++kl) {
int idx_kl = (jF - nsMinF) * nZnT + kl;
gConEff[idx_kl] = (rCon[idx_kl] - rCon0[idx_kl]) * ruFull[idx_kl] +
(zCon[idx_kl] - zCon0[idx_kl]) * zuFull[idx_kl];
}
}
}

inline void AddConstraintForces(
const double* rCon, const double* rCon0, const double* zCon,
const double* zCon0, const double* ruFull, const double* zuFull,
const double* gCon, const double* sqrtSF, int nZnT, int nsMinF, int nsMinF1,
int nsMaxF, double* brmn_e, double* brmn_o, double* bzmn_e, double* bzmn_o,
double* frcon_e, double* frcon_o, double* fzcon_e, double* fzcon_o) {
for (int jF = nsMinF; jF < nsMaxF; ++jF) {
for (int kl = 0; kl < nZnT; ++kl) {
int idx_kl = (jF - nsMinF) * nZnT + kl;

double brcon = (rCon[idx_kl] - rCon0[idx_kl]) * gCon[idx_kl];
double bzcon = (zCon[idx_kl] - zCon0[idx_kl]) * gCon[idx_kl];

brmn_e[idx_kl] += brcon;
bzmn_e[idx_kl] += bzcon;
brmn_o[idx_kl] += brcon * sqrtSF[jF - nsMinF1];
bzmn_o[idx_kl] += bzcon * sqrtSF[jF - nsMinF1];

frcon_e[idx_kl] = ruFull[idx_kl] * gCon[idx_kl];
fzcon_e[idx_kl] = zuFull[idx_kl] * gCon[idx_kl];
frcon_o[idx_kl] = frcon_e[idx_kl] * sqrtSF[jF - nsMinF1];
fzcon_o[idx_kl] = fzcon_e[idx_kl] * sqrtSF[jF - nsMinF1];
}
}
}

inline void ComputeDeAliasConstraintForce(
const double* gConEff, const double* faccon, const double* tcon,
const double* sinmui, const double* cosmui, const double* cosnv,
const double* sinnv, const double* sinmu, const double* cosmu, int nsMinF,
int nsMaxF, int nZeta, int nThetaEff, int nThetaReduced, int nThetaEven,
int mpol, int ntor, int nnyq2, bool lasym, double* m_gsc, double* m_gcs,
double* m_gcc, double* m_gss, double* m_gConAsym, double* m_refl,
double* m_gCon) {
for (int i = 0; i < (nsMaxF - nsMinF) * nZeta * nThetaEff; ++i) {
m_gCon[i] = 0.0;
}
const int jMin = (nsMinF == 0) ? 1 : 0;
for (int jF = (jMin > nsMinF ? jMin : nsMinF); jF < nsMaxF; ++jF) {
if (lasym) {
for (int i = 0; i < nZeta * nThetaEff; ++i) {
m_gConAsym[i] = 0.0;
}
}
for (int m = 1; m < mpol - 1; ++m) {
for (int n = 0; n < ntor + 1; ++n) {
m_gsc[n] = 0.0;
m_gcs[n] = 0.0;
if (lasym) {
m_gcc[n] = 0.0;
m_gss[n] = 0.0;
}
}
for (int k = 0; k < nZeta; ++k) {
const int kl_base = ((jF - nsMinF) * nZeta + k) * nThetaEff;
const int ml_base = m * nThetaReduced;
double w0 = 0.0;
double w1 = 0.0;
for (int l = 0; l < nThetaReduced; ++l) {
w0 += gConEff[kl_base + l] * sinmui[ml_base + l];
w1 += gConEff[kl_base + l] * cosmui[ml_base + l];
}
const double tc = tcon[jF - nsMinF];
if (!lasym) {
for (int n = 0; n < ntor + 1; ++n) {
const int idx_kn = k * (nnyq2 + 1) + n;
m_gsc[n] += cosnv[idx_kn] * w0 * tc;
m_gcs[n] += sinnv[idx_kn] * w1 * tc;
}
} else {
const int k_rev = (nZeta - k) % nZeta;
const int refl_base = ((jF - nsMinF) * nZeta + k_rev) * nThetaEff;
for (int l = 0; l < nThetaReduced; ++l) {
const int l_rev = (nThetaEven - l) % nThetaEven;
m_refl[l] = gConEff[refl_base + l_rev];
}
double w3 = 0.0;
double w4 = 0.0;
for (int l = 0; l < nThetaReduced; ++l) {
w3 += m_refl[l] * cosmui[ml_base + l];
w4 += m_refl[l] * sinmui[ml_base + l];
}
for (int n = 0; n < ntor + 1; ++n) {
const int idx_kn = k * (nnyq2 + 1) + n;
const double cosnv_value = cosnv[idx_kn];
const double sinnv_value = sinnv[idx_kn];
m_gcs[n] += 0.5 * tc * sinnv_value * (w1 - w3);
m_gsc[n] += 0.5 * tc * cosnv_value * (w0 - w4);
m_gss[n] += 0.5 * tc * sinnv_value * (w0 + w4);
m_gcc[n] += 0.5 * tc * cosnv_value * (w1 + w3);
}
}
}

for (int k = 0; k < nZeta; ++k) {
const int kn_base = k * (nnyq2 + 1);
double w0 = 0.0;
double w1 = 0.0;
double a0 = 0.0;
double a1 = 0.0;
for (int n = 0; n < ntor + 1; ++n) {
w0 += m_gsc[n] * cosnv[kn_base + n];
w1 += m_gcs[n] * sinnv[kn_base + n];
if (lasym) {
a0 += m_gcc[n] * cosnv[kn_base + n];
a1 += m_gss[n] * sinnv[kn_base + n];
}
}
for (int l = 0; l < nThetaReduced; ++l) {
const int idx_kl = ((jF - nsMinF) * nZeta + k) * nThetaEff + l;
const int idx_ml = m * nThetaReduced + l;
m_gCon[idx_kl] +=
faccon[m] * (w0 * sinmu[idx_ml] + w1 * cosmu[idx_ml]);
if (lasym) {
const int within = k * nThetaEff + l;
m_gConAsym[within] +=
faccon[m] * (a0 * cosmu[idx_ml] + a1 * sinmu[idx_ml]);
}
}
}
}
if (lasym) {
const int surface_base = (jF - nsMinF) * nZeta * nThetaEff;
for (int k = 0; k < nZeta; ++k) {
const int k_rev = (nZeta - k) % nZeta;
for (int l = nThetaReduced; l < nThetaEven; ++l) {
const int target = surface_base + k * nThetaEff + l;
const int reflected = k_rev * nThetaEff + (nThetaEven - l);
m_gCon[target] =
-m_gCon[surface_base + reflected] + m_gConAsym[reflected];
}
}
for (int k = 0; k < nZeta; ++k) {
for (int l = 0; l < nThetaReduced; ++l) {
const int within = k * nThetaEff + l;
m_gCon[surface_base + within] += m_gConAsym[within];
}
}
}
}
}

inline void ComputeDeAliasConstraintForce(
const double* gConEff, const double* faccon, const double* tcon,
const double* sinmui, const double* cosmui, const double* cosnv,
const double* sinnv, const double* sinmu, const double* cosmu, int nsMinF,
int nsMaxF, int nZeta, int nThetaEff, int nThetaReduced, int mpol, int ntor,
int nnyq2, double* m_gsc, double* m_gcs, double* m_gCon) {
ComputeDeAliasConstraintForce(
gConEff, faccon, tcon, sinmui, cosmui, cosnv, sinnv, sinmu, cosmu, nsMinF,
nsMaxF, nZeta, nThetaEff, nThetaReduced, 2 * (nThetaReduced - 1), mpol,
ntor, nnyq2, false, m_gsc, m_gcs, nullptr, nullptr, nullptr, nullptr,
m_gCon);
}

}

#endif

// header: vmecpp/vmec/ideal_mhd_model/jacobian_kernel.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_JACOBIAN_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_JACOBIAN_KERNEL_H_

namespace vmecpp {

inline void ComputeHalfGridJacobian(
const double* __restrict r1e, const double* __restrict r1o,
const double* __restrict z1e, const double* __restrict z1o,
const double* __restrict rue, const double* __restrict ruo,
const double* __restrict zue, const double* __restrict zuo,
const double* __restrict sqrtSH, double deltaS, double dSHalfDsInterp,
int nZnT, int nsMinF1, int nsMinH, int nsMaxH, double* __restrict r12,
double* __restrict ru12, double* __restrict zu12, double* __restrict rs,
double* __restrict zs, double* __restrict tau) {
for (int jH = nsMinH; jH < nsMaxH; ++jH) {
const double sH = sqrtSH[jH - nsMinH];
for (int kl = 0; kl < nZnT; ++kl) {
const int i_in = (jH - nsMinF1) * nZnT + kl;
const int i_out = (jH + 1 - nsMinF1) * nZnT + kl;
const int ih = (jH - nsMinH) * nZnT + kl;

const double r1e_i = r1e[i_in], r1e_o = r1e[i_out];
const double r1o_i = r1o[i_in], r1o_o = r1o[i_out];
const double z1e_i = z1e[i_in], z1e_o = z1e[i_out];
const double z1o_i = z1o[i_in], z1o_o = z1o[i_out];
const double rue_i = rue[i_in], rue_o = rue[i_out];
const double ruo_i = ruo[i_in], ruo_o = ruo[i_out];
const double zue_i = zue[i_in], zue_o = zue[i_out];
const double zuo_i = zuo[i_in], zuo_o = zuo[i_out];

r12[ih] = 0.5 * ((r1e_i + r1e_o) + sH * (r1o_i + r1o_o));
ru12[ih] = 0.5 * ((rue_i + rue_o) + sH * (ruo_i + ruo_o));
zu12[ih] = 0.5 * ((zue_i + zue_o) + sH * (zuo_i + zuo_o));
rs[ih] = ((r1e_o - r1e_i) + sH * (r1o_o - r1o_i)) / deltaS;
zs[ih] = ((z1e_o - z1e_i) + sH * (z1o_o - z1o_i)) / deltaS;

const double tau1 = ru12[ih] * zs[ih] - rs[ih] * zu12[ih];
const double tau2 =
ruo_o * z1o_o + ruo_i * z1o_i - zuo_o * r1o_o - zuo_i * r1o_i +
(rue_o * z1o_o + rue_i * z1o_i - zue_o * r1o_o - zue_i * r1o_i) / sH;
tau[ih] = tau1 + dSHalfDsInterp * tau2;
}
}
}

}

#endif

// header: vmecpp/vmec/ideal_mhd_model/lambda_force_kernel.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_LAMBDA_FORCE_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_LAMBDA_FORCE_KERNEL_H_

namespace vmecpp {

inline void ComputeHybridLambdaForce(
const double* bsubu, const double* bsubv, const double* gvv,
const double* gsqrt, const double* guv, const double* bsupu,
const double* lu_e, const double* lu_o, const double* sqrtSH,
const double* sqrtSF, const double* radialBlending, double lamscale,
bool lthreed, int nZnT, int nsMinF, int nsMinF1, int nsMinH, int nsMaxH,
int nsMaxFIncludingLcfs, double* bsubu_i, double* bsubv_i,
double* gvv_gsqrt_i, double* guv_bsupu_i, double* blmn_e, double* blmn_o,
double* clmn_e, double* clmn_o) {

int j0 = nsMinF;
double sqrtSHi = 0.0;
if (j0 > 0) {
sqrtSHi = sqrtSH[j0 - 1 - nsMinH];
}
for (int kl = 0; kl < nZnT; ++kl) {
if (j0 == 0) {

bsubu_i[kl] = 0.0;
bsubv_i[kl] = 0.0;
gvv_gsqrt_i[kl] = 0.0;
guv_bsupu_i[kl] = 0.0;
} else {
int iHalf = (j0 - 1 - nsMinH) * nZnT + kl;
bsubu_i[kl] = bsubu[iHalf];
bsubv_i[kl] = bsubv[iHalf];
gvv_gsqrt_i[kl] = gvv[iHalf] / gsqrt[iHalf];
if (lthreed) {
guv_bsupu_i[kl] = guv[iHalf] * bsupu[iHalf];
}
}
}

for (int jF = nsMinF; jF < nsMaxFIncludingLcfs; ++jF) {
double sqrtSHo = 0.0;
if (jF < nsMaxH) {
sqrtSHo = sqrtSH[jF - nsMinH];
}

for (int kl = 0; kl < nZnT; ++kl) {

double bsubv_o = 0.0;
double gvv_gsqrt_o = 0.0;
double guv_bsupu_o = 0.0;
if (jF < nsMaxH) {
int iHalf = (jF - nsMinH) * nZnT + kl;
bsubv_o = bsubv[iHalf];
gvv_gsqrt_o = gvv[iHalf] / gsqrt[iHalf];
if (lthreed) {
guv_bsupu_o = guv[iHalf] * bsupu[iHalf];
}
}

double gvv_gsqrt_lu_e = 0.5 * (gvv_gsqrt_i[kl] + gvv_gsqrt_o) *
lu_e[(jF - nsMinF1) * nZnT + kl];
double gvv_gsqrt_lu_o =
0.5 * (gvv_gsqrt_i[kl] * sqrtSHi + gvv_gsqrt_o * sqrtSHo) *
lu_o[(jF - nsMinF1) * nZnT + kl];

double gvv_gsqrt_lu = gvv_gsqrt_lu_e + gvv_gsqrt_lu_o;
double bsubv_alternative = gvv_gsqrt_lu;
if (lthreed) {
double guv_bsupu = 0.5 * (guv_bsupu_i[kl] + guv_bsupu_o);
bsubv_alternative += guv_bsupu;
}

const double bsubv_average = 0.5 * (bsubv_o + bsubv_i[kl]);

double _blmn = bsubv_average * (1.0 - radialBlending[jF - nsMinF1]) +
bsubv_alternative * radialBlending[jF - nsMinF1];

if (jF > 0) {

_blmn *= -lamscale;
}

blmn_e[(jF - nsMinF) * nZnT + kl] = _blmn;
blmn_o[(jF - nsMinF) * nZnT + kl] = _blmn * sqrtSF[jF - nsMinF1];

if (lthreed) {
double bsubu_o = 0.0;
if (jF < nsMaxH) {
bsubu_o = bsubu[(jF - nsMinH) * nZnT + kl];
}

double _clmn = 0.5 * (bsubu_o + bsubu_i[kl]);

if (jF > 0) {
_clmn *= -lamscale;
}

clmn_e[(jF - nsMinF) * nZnT + kl] = _clmn;
clmn_o[(jF - nsMinF) * nZnT + kl] = _clmn * sqrtSF[jF - nsMinF1];

bsubu_i[kl] = bsubu_o;
}

bsubv_i[kl] = bsubv_o;
gvv_gsqrt_i[kl] = gvv_gsqrt_o;
if (lthreed) {
guv_bsupu_i[kl] = guv_bsupu_o;
}
}
sqrtSHi = sqrtSHo;
}
}

}

#endif

// header: vmecpp/vmec/ideal_mhd_model/metric_kernel.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_METRIC_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_METRIC_KERNEL_H_

namespace vmecpp {

inline void ComputeMetricElements(
const double* __restrict r1e, const double* __restrict r1o,
const double* __restrict rue, const double* __restrict ruo,
const double* __restrict zue, const double* __restrict zuo,
const double* __restrict rve, const double* __restrict rvo,
const double* __restrict zve, const double* __restrict zvo,
const double* __restrict tau, const double* __restrict r12,
const double* __restrict sqrtSF, const double* __restrict sqrtSH,
bool lthreed, int nZnT, int nsMinF1, int nsMinH, int nsMaxH,
double* __restrict gsqrt, double* __restrict guu, double* __restrict guv,
double* __restrict gvv) {
for (int jH = nsMinH; jH < nsMaxH; ++jH) {
const double sF_i = sqrtSF[jH - nsMinF1] * sqrtSF[jH - nsMinF1];
const double sF_o = sqrtSF[jH + 1 - nsMinF1] * sqrtSF[jH + 1 - nsMinF1];
const double sH = sqrtSH[jH - nsMinH];
for (int kl = 0; kl < nZnT; ++kl) {
const int i_in = (jH - nsMinF1) * nZnT + kl;
const int i_out = (jH + 1 - nsMinF1) * nZnT + kl;
const int ih = (jH - nsMinH) * nZnT + kl;

const double r1e_i = r1e[i_in], r1e_o = r1e[i_out];
const double r1o_i = r1o[i_in], r1o_o = r1o[i_out];
const double rue_i = rue[i_in], rue_o = rue[i_out];
const double ruo_i = ruo[i_in], ruo_o = ruo[i_out];
const double zue_i = zue[i_in], zue_o = zue[i_out];
const double zuo_i = zuo[i_in], zuo_o = zuo[i_out];

gsqrt[ih] = tau[ih] * r12[ih];

guu[ih] = 0.5 * ((rue_i * rue_i + zue_i * zue_i) +
(rue_o * rue_o + zue_o * zue_o) +
sF_i * (ruo_i * ruo_i + zuo_i * zuo_i) +
sF_o * (ruo_o * ruo_o + zuo_o * zuo_o)) +
sH * ((rue_i * ruo_i + zue_i * zuo_i) +
(rue_o * ruo_o + zue_o * zuo_o));

gvv[ih] = 0.5 * (r1e_i * r1e_i + r1e_o * r1e_o + sF_i * r1o_i * r1o_i +
sF_o * r1o_o * r1o_o) +
sH * (r1e_i * r1o_i + r1e_o * r1o_o);

if (lthreed) {
const double rve_i = rve[i_in], rve_o = rve[i_out];
const double rvo_i = rvo[i_in], rvo_o = rvo[i_out];
const double zve_i = zve[i_in], zve_o = zve[i_out];
const double zvo_i = zvo[i_in], zvo_o = zvo[i_out];

guv[ih] = 0.5 * ((rue_i * rve_i + zue_i * zve_i) +
(rue_o * rve_o + zue_o * zve_o) +
sF_i * (ruo_i * rvo_i + zuo_i * zvo_i) +
sF_o * (ruo_o * rvo_o + zuo_o * zvo_o) +
sH * ((rue_i * rvo_i + zue_i * zvo_i) +
(rue_o * rvo_o + zue_o * zvo_o) +
(rve_i * ruo_i + zve_i * zuo_i) +
(rve_o * ruo_o + zve_o * zuo_o)));

gvv[ih] += 0.5 * ((rve_i * rve_i + zve_i * zve_i) +
(rve_o * rve_o + zve_o * zve_o) +
sF_i * (rvo_i * rvo_i + zvo_i * zvo_i) +
sF_o * (rvo_o * rvo_o + zvo_o * zvo_o)) +
sH * ((rve_i * rvo_i + zve_i * zvo_i) +
(rve_o * rvo_o + zve_o * zvo_o));
}
}
}
}

}

#endif

// header: vmecpp/vmec/ideal_mhd_model/mhdforce_kernel.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_MHDFORCE_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_MHDFORCE_KERNEL_H_

#include <Eigen/Dense>

namespace vmecpp {

inline void ComputeMHDForceDensity(

const double* r1_e, const double* r1_o, const double* ru_e,
const double* ru_o, const double* zu_e, const double* zu_o,
const double* z1_o, const double* rv_e, const double* rv_o,
const double* zv_e, const double* zv_o,

const double* r12, const double* ru12, const double* zu12, const double* rs,
const double* zs, const double* tau, const double* totalPressure,
const double* gsqrt, const double* bsupu, const double* bsupv,

const double* sqrtSF, const double* sqrtSH,

double* P_i, double* rup_i, double* zup_i, double* rsp_i, double* zsp_i,
double* taup_i, double* gbubu_i, double* gbubv_i, double* gbvbv_i,

double* P_o, double* rup_o, double* zup_o, double* rsp_o, double* zsp_o,
double* taup_o, double* gbubu_o, double* gbubv_o, double* gbvbv_o,
double* P_avg, double* P_wavg, double* gbubu_avg, double* gbubu_wavg,
double* gbvbv_avg, double* gbvbv_wavg, double* gbubv_avg,
double* gbubv_wavg,

double deltaS, int nZnT, int nsMinF, int nsMinF1, int nsMinH, int nsMaxH,
int jMaxRZ, bool lthreed,

double* armn_e, double* armn_o, double* azmn_e, double* azmn_o,
double* brmn_e, double* brmn_o, double* bzmn_e, double* bzmn_o,
double* crmn_e, double* crmn_o, double* czmn_e, double* czmn_o) {
using V = Eigen::VectorXd;
using Map = Eigen::Map<V>;
using CMap = Eigen::Map<const V>;

Map vP_i(P_i, nZnT), vrup_i(rup_i, nZnT), vzup_i(zup_i, nZnT);
Map vrsp_i(rsp_i, nZnT), vzsp_i(zsp_i, nZnT), vtaup_i(taup_i, nZnT);
Map vgbubu_i(gbubu_i, nZnT), vgbubv_i(gbubv_i, nZnT), vgbvbv_i(gbvbv_i, nZnT);
Map vP_o(P_o, nZnT), vrup_o(rup_o, nZnT), vzup_o(zup_o, nZnT);
Map vrsp_o(rsp_o, nZnT), vzsp_o(zsp_o, nZnT), vtaup_o(taup_o, nZnT);
Map vgbubu_o(gbubu_o, nZnT), vgbubv_o(gbubv_o, nZnT), vgbvbv_o(gbvbv_o, nZnT);
Map vP_avg(P_avg, nZnT), vP_wavg(P_wavg, nZnT);
Map vgbubu_avg(gbubu_avg, nZnT), vgbubu_wavg(gbubu_wavg, nZnT);
Map vgbvbv_avg(gbvbv_avg, nZnT), vgbvbv_wavg(gbvbv_wavg, nZnT);
Map vgbubv_avg(gbubv_avg, nZnT), vgbubv_wavg(gbubv_wavg, nZnT);

double sqrtSHi = 1.0;
if (nsMinF > 0) {
const int j0 = nsMinH;
for (int kl = 0; kl < nZnT; ++kl) {
const int iHalf = (j0 - nsMinH) * nZnT + kl;
P_i[kl] = r12[iHalf] * totalPressure[iHalf];
rup_i[kl] = ru12[iHalf] * P_i[kl];
zup_i[kl] = zu12[iHalf] * P_i[kl];
rsp_i[kl] = rs[iHalf] * P_i[kl];
zsp_i[kl] = zs[iHalf] * P_i[kl];
taup_i[kl] = tau[iHalf] * totalPressure[iHalf];
gbubu_i[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupu[iHalf];
gbubv_i[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupv[iHalf];
gbvbv_i[kl] = gsqrt[iHalf] * bsupv[iHalf] * bsupv[iHalf];
}
sqrtSHi = sqrtSH[j0 - nsMinH];
} else {
vP_i.setZero();
vrup_i.setZero();
vzup_i.setZero();
vrsp_i.setZero();
vzsp_i.setZero();
vtaup_i.setZero();
vgbubu_i.setZero();
vgbubv_i.setZero();
vgbvbv_i.setZero();
}

vP_o.setZero();
vrup_o.setZero();
vzup_o.setZero();
vrsp_o.setZero();
vzsp_o.setZero();
vtaup_o.setZero();
vgbubu_o.setZero();
vgbubv_o.setZero();
vgbvbv_o.setZero();

for (int jF = nsMinF; jF < jMaxRZ; ++jF) {
const double sFull = sqrtSF[jF - nsMinF1] * sqrtSF[jF - nsMinF1];
double sqrtSHo = 1.0;
if (jF < nsMaxH) {
sqrtSHo = sqrtSH[jF - nsMinH];
}

if (jF < nsMaxH) {
const int iHalf_base = (jF - nsMinH) * nZnT;
for (int kl = 0; kl < nZnT; ++kl) {
const int iHalf = iHalf_base + kl;
P_o[kl] = r12[iHalf] * totalPressure[iHalf];
rup_o[kl] = ru12[iHalf] * P_o[kl];
zup_o[kl] = zu12[iHalf] * P_o[kl];
rsp_o[kl] = rs[iHalf] * P_o[kl];
zsp_o[kl] = zs[iHalf] * P_o[kl];
taup_o[kl] = tau[iHalf] * totalPressure[iHalf];
}
for (int kl = 0; kl < nZnT; ++kl) {
const int iHalf = iHalf_base + kl;
gbubu_o[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupu[iHalf];
gbubv_o[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupv[iHalf];
gbvbv_o[kl] = gsqrt[iHalf] * bsupv[iHalf] * bsupv[iHalf];
}
} else {
vP_o.setZero();
vrup_o.setZero();
vzup_o.setZero();
vrsp_o.setZero();
vzsp_o.setZero();
vtaup_o.setZero();
vgbubu_o.setZero();
vgbubv_o.setZero();
vgbvbv_o.setZero();
}

const int g_off = (jF - nsMinF1) * nZnT;
const int f_off = (jF - nsMinF) * nZnT;
const CMap r1e(r1_e + g_off, nZnT), r1o(r1_o + g_off, nZnT);
const CMap rue(ru_e + g_off, nZnT), ruo(ru_o + g_off, nZnT);
const CMap zue(zu_e + g_off, nZnT), zuo(zu_o + g_off, nZnT);
const CMap z1o(z1_o + g_off, nZnT);

const double invDS = 1.0 / deltaS;
const double invSHo = 1.0 / sqrtSHo;
const double invSHi = 1.0 / sqrtSHi;
vP_avg = 0.5 * (vP_o + vP_i);
vP_wavg = 0.5 * (vP_o * invSHo + vP_i * invSHi);
vgbubu_avg = 0.5 * (vgbubu_o + vgbubu_i);
vgbubu_wavg = 0.5 * (vgbubu_o * sqrtSHo + vgbubu_i * sqrtSHi);
vgbvbv_avg = 0.5 * (vgbvbv_o + vgbvbv_i);
vgbvbv_wavg = 0.5 * (vgbvbv_o * sqrtSHo + vgbvbv_i * sqrtSHi);

Map(armn_e + f_off, nZnT) =
(vzup_o - vzup_i) * invDS + 0.5 * (vtaup_o + vtaup_i) -
vgbvbv_avg.cwiseProduct(r1e) - vgbvbv_wavg.cwiseProduct(r1o);
Map(armn_o + f_off, nZnT) =
(vzup_o * sqrtSHo - vzup_i * sqrtSHi) * invDS -
0.5 * vP_wavg.cwiseProduct(zue) - 0.5 * vP_avg.cwiseProduct(zuo) +
0.5 * (vtaup_o * sqrtSHo + vtaup_i * sqrtSHi) -
vgbvbv_wavg.cwiseProduct(r1e) - vgbvbv_avg.cwiseProduct(r1o) * sFull;

Map(azmn_e + f_off, nZnT) = -(vrup_o - vrup_i) * invDS;
Map(azmn_o + f_off, nZnT) = -(vrup_o * sqrtSHo - vrup_i * sqrtSHi) * invDS +
0.5 * vP_wavg.cwiseProduct(rue) +
0.5 * vP_avg.cwiseProduct(ruo);

Map(brmn_e + f_off, nZnT) =
0.5 * (vzsp_o + vzsp_i) + 0.5 * vP_wavg.cwiseProduct(z1o) -
vgbubu_avg.cwiseProduct(rue) - vgbubu_wavg.cwiseProduct(ruo);
Map(brmn_o + f_off, nZnT) = 0.5 * (vzsp_o * sqrtSHo + vzsp_i * sqrtSHi) +
0.5 * vP_avg.cwiseProduct(z1o) -
vgbubu_wavg.cwiseProduct(rue) -
vgbubu_avg.cwiseProduct(ruo) * sFull;

Map(bzmn_e + f_off, nZnT) =
-0.5 * (vrsp_o + vrsp_i) - 0.5 * vP_wavg.cwiseProduct(r1o) -
vgbubu_avg.cwiseProduct(zue) - vgbubu_wavg.cwiseProduct(zuo);
Map(bzmn_o + f_off, nZnT) = -0.5 * (vrsp_o * sqrtSHo + vrsp_i * sqrtSHi) -
0.5 * vP_avg.cwiseProduct(r1o) -
vgbubu_wavg.cwiseProduct(zue) -
vgbubu_avg.cwiseProduct(zuo) * sFull;

if (lthreed) {
vgbubv_avg = 0.5 * (vgbubv_o + vgbubv_i);
vgbubv_wavg = 0.5 * (vgbubv_o * sqrtSHo + vgbubv_i * sqrtSHi);
const CMap rve(rv_e + g_off, nZnT), rvo(rv_o + g_off, nZnT);
const CMap zve(zv_e + g_off, nZnT), zvo(zv_o + g_off, nZnT);

Map(brmn_e + f_off, nZnT) -=
vgbubv_avg.cwiseProduct(rve) + vgbubv_wavg.cwiseProduct(rvo);
Map(brmn_o + f_off, nZnT) -=
vgbubv_wavg.cwiseProduct(rve) + vgbubv_avg.cwiseProduct(rvo) * sFull;
Map(bzmn_e + f_off, nZnT) -=
vgbubv_avg.cwiseProduct(zve) + vgbubv_wavg.cwiseProduct(zvo);
Map(bzmn_o + f_off, nZnT) -=
vgbubv_wavg.cwiseProduct(zve) + vgbubv_avg.cwiseProduct(zvo) * sFull;

Map(crmn_e + f_off, nZnT) =
vgbubv_avg.cwiseProduct(rue) + vgbubv_wavg.cwiseProduct(ruo) +
vgbvbv_avg.cwiseProduct(rve) + vgbvbv_wavg.cwiseProduct(rvo);
Map(crmn_o + f_off, nZnT) =
vgbubv_wavg.cwiseProduct(rue) + vgbubv_avg.cwiseProduct(ruo) * sFull +
vgbvbv_wavg.cwiseProduct(rve) + vgbvbv_avg.cwiseProduct(rvo) * sFull;

Map(czmn_e + f_off, nZnT) =
vgbubv_avg.cwiseProduct(zue) + vgbubv_wavg.cwiseProduct(zuo) +
vgbvbv_avg.cwiseProduct(zve) + vgbvbv_wavg.cwiseProduct(zvo);
Map(czmn_o + f_off, nZnT) =
vgbubv_wavg.cwiseProduct(zue) + vgbubv_avg.cwiseProduct(zuo) * sFull +
vgbvbv_wavg.cwiseProduct(zve) + vgbvbv_avg.cwiseProduct(zvo) * sFull;
}

vP_i = vP_o;
vrup_i = vrup_o;
vzup_i = vzup_o;
vrsp_i = vrsp_o;
vzsp_i = vzsp_o;
vtaup_i = vtaup_o;
vgbubu_i = vgbubu_o;
vgbubv_i = vgbubv_o;
vgbvbv_i = vgbvbv_o;
sqrtSHi = sqrtSHo;
}
}

}

#endif

// header: vmecpp/vmec/ideal_mhd_model/pressure_kernel.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_PRESSURE_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_PRESSURE_KERNEL_H_

namespace vmecpp {

inline void ComputeMagneticPressure(const double* bsupu, const double* bsubu,
const double* bsupv, const double* bsubv,
int n, double* total_pressure) {
for (int i = 0; i < n; ++i) {
total_pressure[i] = 0.5 * (bsupu[i] * bsubu[i] + bsupv[i] * bsubv[i]);
}
}

}

#endif

namespace vmecpp {

struct LocalForceComposition {
int nZnT;
int geom_stride;
int force_stride;

int nsMinF, nsMinF1, nsMinH, nsMaxH;
int jMaxRZ;
int nsMaxFIncludingLcfs;

const double* sqrtSF;
const double* sqrtSH;
const double* chipH;
const double* presH;
const double* radialBlending;
double deltaS;
double dSHalfDsInterp;
double lamscale;
bool lthreed;

int ncurr = 0;
int nThetaEff = 0;
const double* currH = nullptr;
const double* wInt = nullptr;

bool with_constraint = false;
bool lasym = false;
int ns = 0;
int nsMaxF = 0;
int nZeta = 0, nThetaEven = 0, nThetaReduced = 0, mpol = 0, ntor = 0,
nnyq2 = 0;
double tcon_multiplier = 0.0;
const double* rCon0 = nullptr;
const double* zCon0 = nullptr;
const double* faccon = nullptr;
const double* sinmui = nullptr;
const double* cosmui = nullptr;
const double* cosnv = nullptr;
const double* sinnv = nullptr;
const double* sinmu = nullptr;
const double* cosmu = nullptr;
};

inline constexpr int kLocalForceBlocks = 21;

inline int LocalForceWorkSize(const LocalForceComposition& c) {
const int nHalf = c.nsMaxH - c.nsMinH;
int n = 15 * nHalf * c.nZnT + 30 * c.nZnT;
if (c.with_constraint) {
const int nFull = c.nsMaxFIncludingLcfs - c.nsMinF;
n += 4 * nFull * c.nZnT + 4 * (c.ntor + 1) + c.nZnT + c.nThetaReduced +
nFull + 2 * nHalf;
}
return n;
}

inline void ComputeLocalForceDensity(const double* geom, double* work,
double* force,
const LocalForceComposition* c) {
const int nZnT = c->nZnT;
const int gS = c->geom_stride;
const int fS = c->force_stride;
const int nH = (c->nsMaxH - c->nsMinH) * nZnT;
const double* r1e = geom + 0 * gS;
const double* r1o = geom + 1 * gS;
const double* z1e = geom + 2 * gS;
const double* z1o = geom + 3 * gS;
const double* rue = geom + 4 * gS;
const double* ruo = geom + 5 * gS;
const double* zue = geom + 6 * gS;
const double* zuo = geom + 7 * gS;
const double* rve = geom + 8 * gS;
const double* rvo = geom + 9 * gS;
const double* zve = geom + 10 * gS;
const double* zvo = geom + 11 * gS;
const double* lue = geom + 12 * gS;
const double* luo = geom + 13 * gS;
const double* lve = geom + 14 * gS;
const double* lvo = geom + 15 * gS;

double* p = work;
double* r12 = p;
p += nH;
double* ru12 = p;
p += nH;
double* zu12 = p;
p += nH;
double* rs = p;
p += nH;
double* zs = p;
p += nH;
double* tau = p;
p += nH;
double* gsqrt = p;
p += nH;
double* guu = p;
p += nH;
double* guv = p;
p += nH;
double* gvv = p;
p += nH;
double* bsupu = p;
p += nH;
double* bsupv = p;
p += nH;
double* bsubu = p;
p += nH;
double* bsubv = p;
p += nH;
double* tp = p;
p += nH;
double* s = p;

ComputeHalfGridJacobian(r1e, r1o, z1e, z1o, rue, ruo, zue, zuo, c->sqrtSH,
c->deltaS, c->dSHalfDsInterp, nZnT, c->nsMinF1,
c->nsMinH, c->nsMaxH, r12, ru12, zu12, rs, zs, tau);
ComputeMetricElements(r1e, r1o, rue, ruo, zue, zuo, rve, rvo, zve, zvo, tau,
r12, c->sqrtSF, c->sqrtSH, c->lthreed, nZnT, c->nsMinF1,
c->nsMinH, c->nsMaxH, gsqrt, guu, guv, gvv);
ComputeBsupContra(lue, luo, lve, lvo, gsqrt, c->sqrtSH, c->lthreed, nZnT,
c->nsMinF1, c->nsMinH, c->nsMaxH, bsupu, bsupv);
double* chip_out = force + 20 * fS;
for (int jH = c->nsMinH; jH < c->nsMaxH; ++jH) {

double chip = c->chipH[jH - c->nsMinH];
if (c->ncurr == 1) {
double jvPlasma = 0.0;
double avg_guu_gsqrt = 0.0;
for (int kl = 0; kl < nZnT; ++kl) {
const int ih = (jH - c->nsMinH) * nZnT + kl;
const int l = kl % c->nThetaEff;
if (c->lthreed) {
jvPlasma += (guu[ih] * bsupu[ih] + guv[ih] * bsupv[ih]) * c->wInt[l];
} else {
jvPlasma += guu[ih] * bsupu[ih] * c->wInt[l];
}
avg_guu_gsqrt += guu[ih] / gsqrt[ih] * c->wInt[l];
}
if (avg_guu_gsqrt != 0.0) {
chip = (c->currH[jH - c->nsMinH] - jvPlasma) / avg_guu_gsqrt;
}

chip_out[jH - c->nsMinH] = chip;
}
for (int kl = 0; kl < nZnT; ++kl) {
const int ih = (jH - c->nsMinH) * nZnT + kl;
bsupu[ih] += chip / gsqrt[ih];
}
}
ComputeBCo(guu, guv, gvv, bsupu, bsupv, c->lthreed, nH, bsubu, bsubv);
ComputeMagneticPressure(bsupu, bsubu, bsupv, bsubv, nH, tp);
for (int jH = c->nsMinH; jH < c->nsMaxH; ++jH) {
for (int kl = 0; kl < nZnT; ++kl)
tp[(jH - c->nsMinH) * nZnT + kl] += c->presH[jH - c->nsMinH];
}

double* P_i = s;
s += nZnT;
double* rup_i = s;
s += nZnT;
double* zup_i = s;
s += nZnT;
double* rsp_i = s;
s += nZnT;
double* zsp_i = s;
s += nZnT;
double* taup_i = s;
s += nZnT;
double* gbubu_i = s;
s += nZnT;
double* gbubv_i = s;
s += nZnT;
double* gbvbv_i = s;
s += nZnT;
double* P_o = s;
s += nZnT;
double* rup_o = s;
s += nZnT;
double* zup_o = s;
s += nZnT;
double* rsp_o = s;
s += nZnT;
double* zsp_o = s;
s += nZnT;
double* taup_o = s;
s += nZnT;
double* gbubu_o = s;
s += nZnT;
double* gbubv_o = s;
s += nZnT;
double* gbvbv_o = s;
s += nZnT;
double* P_avg = s;
s += nZnT;
double* P_wavg = s;
s += nZnT;
double* gbubu_avg = s;
s += nZnT;
double* gbubu_wavg = s;
s += nZnT;
double* gbvbv_avg = s;
s += nZnT;
double* gbvbv_wavg = s;
s += nZnT;
double* gbubv_avg = s;
s += nZnT;
double* gbubv_wavg = s;
s += nZnT;
double* bsubu_i = s;
s += nZnT;
double* bsubv_i = s;
s += nZnT;
double* gvv_gsqrt_i = s;
s += nZnT;
double* guv_bsupu_i = s;
s += nZnT;

double* armn_e = force + 0 * fS;
double* armn_o = force + 1 * fS;
double* azmn_e = force + 2 * fS;
double* azmn_o = force + 3 * fS;
double* brmn_e = force + 4 * fS;
double* brmn_o = force + 5 * fS;
double* bzmn_e = force + 6 * fS;
double* bzmn_o = force + 7 * fS;
double* crmn_e = force + 8 * fS;
double* crmn_o = force + 9 * fS;
double* czmn_e = force + 10 * fS;
double* czmn_o = force + 11 * fS;
ComputeMHDForceDensity(
r1e, r1o, rue, ruo, zue, zuo, z1o, rve, rvo, zve, zvo, r12, ru12, zu12,
rs, zs, tau, tp, gsqrt, bsupu, bsupv, c->sqrtSF, c->sqrtSH, P_i, rup_i,
zup_i, rsp_i, zsp_i, taup_i, gbubu_i, gbubv_i, gbvbv_i, P_o, rup_o, zup_o,
rsp_o, zsp_o, taup_o, gbubu_o, gbubv_o, gbvbv_o, P_avg, P_wavg, gbubu_avg,
gbubu_wavg, gbvbv_avg, gbvbv_wavg, gbubv_avg, gbubv_wavg, c->deltaS, nZnT,
c->nsMinF, c->nsMinF1, c->nsMinH, c->nsMaxH, c->jMaxRZ, c->lthreed,
armn_e, armn_o, azmn_e, azmn_o, brmn_e, brmn_o, bzmn_e, bzmn_o, crmn_e,
crmn_o, czmn_e, czmn_o);

double* blmn_e = force + 12 * fS;
double* blmn_o = force + 13 * fS;
double* clmn_e = force + 14 * fS;
double* clmn_o = force + 15 * fS;
ComputeHybridLambdaForce(
bsubu, bsubv, gvv, gsqrt, guv, bsupu, lue, luo, c->sqrtSH, c->sqrtSF,
c->radialBlending, c->lamscale, c->lthreed, nZnT, c->nsMinF, c->nsMinF1,
c->nsMinH, c->nsMaxH, c->nsMaxFIncludingLcfs, bsubu_i, bsubv_i,
gvv_gsqrt_i, guv_bsupu_i, blmn_e, blmn_o, clmn_e, clmn_o);

if (c->with_constraint) {

const double* rCon = geom + 16 * gS;
const double* zCon = geom + 17 * gS;
const double* ruFull = geom + 18 * gS;
const double* zuFull = geom + 19 * gS;
double* gConEff = s;
s += (c->nsMaxFIncludingLcfs - c->nsMinF) * nZnT;
double* gCon = s;
s += (c->nsMaxF - c->nsMinF) * nZnT;
double* gsc = s;
s += c->ntor + 1;
double* gcs = s;
s += c->ntor + 1;
double* gcc = s;
s += c->ntor + 1;
double* gss = s;
s += c->ntor + 1;
double* gConAsym = s;
s += nZnT;
double* refl = s;
s += c->nThetaReduced;

double* rCon0 = s;
s += (c->nsMaxFIncludingLcfs - c->nsMinF) * nZnT;
double* zCon0 = s;
s += (c->nsMaxFIncludingLcfs - c->nsMinF) * nZnT;

double* tcon = s;
s += c->nsMaxFIncludingLcfs - c->nsMinF;
double* ard_h = s;
s += c->nsMaxH - c->nsMinH;
double* azd_h = s;
for (int jH = c->nsMinH; jH < c->nsMaxH; ++jH) {
double ar = 0.0;
double az = 0.0;
for (int kl = 0; kl < nZnT; ++kl) {
const int ih = (jH - c->nsMinH) * nZnT + kl;

const double pTau =
-4.0 * r12[ih] * tp[ih] / tau[ih] * c->wInt[kl % c->nThetaEff];
const double zu = zu12[ih] / c->deltaS;
const double ru = ru12[ih] / c->deltaS;
ar += pTau * zu * zu;
az += pTau * ru * ru;
}
ard_h[jH - c->nsMinH] = ar;
azd_h[jH - c->nsMinH] = az;
}
for (int i = 0; i < c->nsMaxFIncludingLcfs - c->nsMinF; ++i) {
tcon[i] = 0.0;
}
const double tcon_scale =
c->tcon_multiplier * 32.0 * c->deltaS * 32.0 * c->deltaS;
for (int jF = (c->nsMinF > 0 ? c->nsMinF : 1); jF < c->nsMaxF; ++jF) {
double arNorm = 0.0;
double azNorm = 0.0;
for (int kl = 0; kl < nZnT; ++kl) {
const int idx = (jF - c->nsMinF) * nZnT + kl;
const double w = c->wInt[kl % c->nThetaEff];
arNorm += ruFull[idx] * ruFull[idx] * w;
azNorm += zuFull[idx] * zuFull[idx] * w;
}
const double ard = ard_h[jF - 1 - c->nsMinH] +
(jF < c->ns - 1 ? ard_h[jF - c->nsMinH] : 0.0);
const double azd = azd_h[jF - 1 - c->nsMinH] +
(jF < c->ns - 1 ? azd_h[jF - c->nsMinH] : 0.0);
tcon[jF - c->nsMinF] =
std::min(std::fabs(ard / arNorm), std::fabs(azd / azNorm)) *
tcon_scale;
}
if (c->nsMaxFIncludingLcfs == c->ns) {

tcon[c->ns - 1 - c->nsMinF] = 0.5 * tcon[c->ns - 2 - c->nsMinF];
}
const int lcfs = (c->nsMaxFIncludingLcfs - 1 - c->nsMinF) * nZnT;
for (int jF = (c->nsMinF > 1 ? c->nsMinF : 1); jF < c->nsMaxFIncludingLcfs;
++jF) {
const double sf = c->sqrtSF[jF - c->nsMinF1] * c->sqrtSF[jF - c->nsMinF1];
for (int kl = 0; kl < nZnT; ++kl) {
const int idx = (jF - c->nsMinF) * nZnT + kl;
rCon0[idx] = rCon[lcfs + kl] * sf;
zCon0[idx] = zCon[lcfs + kl] * sf;
}
}
ComputeEffectiveConstraintForce(rCon, rCon0, zCon, zCon0, ruFull, zuFull,
nZnT, c->nsMinF, c->nsMaxFIncludingLcfs,
gConEff);
ComputeDeAliasConstraintForce(
gConEff, c->faccon, tcon, c->sinmui, c->cosmui, c->cosnv, c->sinnv,
c->sinmu, c->cosmu, c->nsMinF, c->nsMaxF, c->nZeta, c->nThetaEff,
c->nThetaReduced, c->nThetaEven, c->mpol, c->ntor, c->nnyq2, c->lasym,
gsc, gcs, gcc, gss, gConAsym, refl, gCon);
double* frcon_e = force + 16 * fS;
double* frcon_o = force + 17 * fS;
double* fzcon_e = force + 18 * fS;
double* fzcon_o = force + 19 * fS;
AddConstraintForces(rCon, rCon0, zCon, zCon0, ruFull, zuFull, gCon,
c->sqrtSF, nZnT, c->nsMinF, c->nsMinF1, c->nsMaxF,
brmn_e, brmn_o, bzmn_e, bzmn_o, frcon_e, frcon_o,
fzcon_e, fzcon_o);
}
}

}

#endif

// header: vmecpp/vmec/thread_local_storage/thread_local_storage.h

#ifndef VMECPP_VMEC_THREAD_LOCAL_STORAGE_THREAD_LOCAL_STORAGE_H_
#define VMECPP_VMEC_THREAD_LOCAL_STORAGE_THREAD_LOCAL_STORAGE_H_

#include <Eigen/Dense>

namespace vmecpp {

class ThreadLocalStorage {
const Sizes& s_;

public:
explicit ThreadLocalStorage(const Sizes* s);

Eigen::VectorXd r1e_i;
Eigen::VectorXd r1o_i;
Eigen::VectorXd rue_i;
Eigen::VectorXd ruo_i;
Eigen::VectorXd rve_i;
Eigen::VectorXd rvo_i;
Eigen::VectorXd z1e_i;
Eigen::VectorXd z1o_i;
Eigen::VectorXd zue_i;
Eigen::VectorXd zuo_i;
Eigen::VectorXd zve_i;
Eigen::VectorXd zvo_i;
Eigen::VectorXd lue_i;
Eigen::VectorXd luo_i;
Eigen::VectorXd lve_i;
Eigen::VectorXd lvo_i;

Eigen::VectorXd bsubu_i;
Eigen::VectorXd bsubv_i;
Eigen::VectorXd gvv_gsqrt_i;
Eigen::VectorXd guv_bsupu_i;

Eigen::VectorXd P_i;
Eigen::VectorXd rup_i;
Eigen::VectorXd zup_i;
Eigen::VectorXd rsp_i;
Eigen::VectorXd zsp_i;
Eigen::VectorXd taup_i;
Eigen::VectorXd gbubu_i;
Eigen::VectorXd gbubv_i;
Eigen::VectorXd gbvbv_i;

Eigen::VectorXd P_o;
Eigen::VectorXd rup_o;
Eigen::VectorXd zup_o;
Eigen::VectorXd rsp_o;
Eigen::VectorXd zsp_o;
Eigen::VectorXd taup_o;
Eigen::VectorXd gbubu_o;
Eigen::VectorXd gbubv_o;
Eigen::VectorXd gbvbv_o;
Eigen::VectorXd P_avg;
Eigen::VectorXd P_wavg;
Eigen::VectorXd gbubu_avg;
Eigen::VectorXd gbubu_wavg;
Eigen::VectorXd gbvbv_avg;
Eigen::VectorXd gbvbv_wavg;
Eigen::VectorXd gbubv_avg;
Eigen::VectorXd gbubv_wavg;
};

}

#endif

namespace vmecpp {

void deAliasConstraintForce(const RadialPartitioning& rp,
const FourierBasisFastPoloidal& fb, const Sizes& s_,
const Eigen::VectorXd& faccon,
const Eigen::VectorXd& tcon,
const Eigen::VectorXd& gConEff,
Eigen::VectorXd& m_gsc, Eigen::VectorXd& m_gcs,
Eigen::VectorXd& m_gCon);
void deAliasConstraintForce(const RadialPartitioning& rp,
const FourierBasisFastPoloidal& fb, const Sizes& s_,
const Eigen::VectorXd& faccon,
const Eigen::VectorXd& tcon,
const Eigen::VectorXd& gConEff,
Eigen::VectorXd& m_gsc, Eigen::VectorXd& m_gcs,
Eigen::VectorXd& m_gcc, Eigen::VectorXd& m_gss,
Eigen::VectorXd& m_gConAsym,
Eigen::VectorXd& m_refl, Eigen::VectorXd& m_gCon);

class IdealMhdModel {
public:
IdealMhdModel(FlowControl* m_fc, const Sizes* s,
const FourierBasisFastPoloidal* t, RadialProfiles* m_p,
const VmecConstants* constants, ThreadLocalStorage* m_ls,
HandoverStorage* m_h, const RadialPartitioning* r,
const std::vector<std::unique_ptr<FreeBoundaryBase>>* m_fb_vac,
int vac_num_threads, int signOfJacobian, int nvacskip,
VacuumPressureState* m_vacuum_pressure_state);

void setFromINDATA(int ncurr, double adiabaticIndex, double tCon0,
bool lforbal);

void evalFResInvar(const Eigen::Vector3d& localFResInvar);

void evalFResPrecd(const Eigen::Vector3d& localFResPrecd);

absl::StatusOr<bool> update(
FourierGeometry& m_decomposed_x, FourierGeometry& m_physical_x,
FourierForces& m_decomposed_f, FourierForces& m_physical_f,
bool& m_need_restart, int& m_last_preconditioner_update,
int& m_last_full_update_nestor, FlowControl& m_fc, const int iter1,
const int iter2, const VmecCheckpoint& checkpoint = VmecCheckpoint::NONE,
const int iterations_before_checkpointing = INT_MAX, bool verbose = true,
bool always_fix_m1_gauge = false);
std::int64_t forceEvaluationCount() const { return force_evaluation_count_; }
void resetForceEvaluationCount() { force_evaluation_count_ = 0; }

void geometryFromFourier(const FourierGeometry& physical_x);

void dft_FourierToReal_3d_symm(const FourierGeometry& physical_x);

void dft_FourierToReal_3d_asymm(const FourierGeometry& physical_x);

void dft_FourierToReal_2d_symm(const FourierGeometry& physical_x);

void dft_FourierToReal_2d_asymm(const FourierGeometry& physical_x);

void symrzl();

void rzConIntoVolume();

void computeJacobian();

void computeMetricElements();

void updateDifferentialVolume();

void computeInitialVolume();

void updateVolume();

void computeBContra();

void computeBCo();

void pressureAndEnergies();

void radialForceBalance();

void hybridLambdaForce();

void computeForceNorms(const FourierGeometry& decomposed_x);

void computeMHDForces();

absl::Status constraintForceMultiplier();

double constraintMultiplierScale() const;

void effectiveConstraintForce();

void deAliasConstraintForce();

void assembleTotalForces();

void forcesToFourier(FourierForces& m_physical_f);

void applyExactForceJacobian(const double* geomP, const double* dgeom,
int geom_stride, FourierForces& m_physical_f,
FourierForces& m_decomposed_hv,
bool fix_m1_gauge);

void packGeometry(FourierGeometry& m_decomposed,
FourierGeometry& m_physical_scratch, double* out, int gS,
bool primal);

double composedForceResidual(const double* geomP, int geom_stride);

void exactForceDensityTangent(const double* geomP, const double* dgeom,
int geom_stride, double* dforce_out);

void exactForceDensityCotangent(const double* geomP, const double* force_bar,
int geom_stride, double* geom_bar_out);

LocalForceComposition makeLocalForceComposition(int geom_stride);

void applyExactForceJacobianTranspose(const double* geomP, int geom_stride,
FourierForces& m_decomposed_in,
FourierForces& m_physical_f,
FourierGeometry& m_physical_scratch,
FourierGeometry& m_decomposed_out,
bool fix_m1_gauge);

void chipStateVjp(const double* geomP, int geom_stride,
const double* chip_bar, FourierGeometry& m_physical_scratch,
FourierGeometry& m_decomposed_out);

void dft_ForcesToFourierTranspose_2d_symm(const FourierForces& m_coeff_bar);
void dft_FourierToRealTranspose_2d_symm(FourierGeometry& m_coeff_bar_out);
void dft_ForcesToFourierTranspose_3d_symm(const FourierForces& m_coeff_bar);
void dft_FourierToRealTranspose_3d_symm(FourierGeometry& m_coeff_bar_out);

void dft_ForcesToFourier_3d_symm(FourierForces& m_physical_f);

void dft_ForcesToFourier_3d_asymm(FourierForces& m_physical_f);

void dft_ForcesToFourier_2d_symm(FourierForces& m_physical_f);

void symforce();

void dft_ForcesToFourier_2d_asymm(FourierForces& m_physical_f);

bool shouldUpdateRadialPreconditioner(int iter1, int iter2) const;

void updateRadialPreconditioner();

void updateLambdaPreconditioner();

void computePreconditioningMatrix(
const Eigen::VectorXd& xs, const Eigen::VectorXd& xu12,
const Eigen::VectorXd& xu_e, const Eigen::VectorXd& xu_o,
const Eigen::VectorXd& x1_o, Eigen::VectorXd& m_axm,
Eigen::VectorXd& m_axd, Eigen::VectorXd& m_bxm, Eigen::VectorXd& m_bxd,
Eigen::VectorXd& m_cxd, const Eigen::VectorXd& trigmult,
Eigen::VectorXd& m_eqfactor);

void applyM1Preconditioner(FourierForces& m_decomposed_f);

void assembleRZPreconditioner();

absl::Status applyRZPreconditioner(FourierForces& m_decomposed_f);

void applyLambdaPreconditioner(FourierForces& m_decomposed_f);

double get_delbsq() const;

int get_ivacskip() const;

Eigen::VectorXd r1_e;
Eigen::VectorXd r1_o;

Eigen::VectorXd ru_e;
Eigen::VectorXd ru_o;

Eigen::VectorXd rv_e;
Eigen::VectorXd rv_o;

Eigen::VectorXd z1_e;
Eigen::VectorXd z1_o;

Eigen::VectorXd zu_e;
Eigen::VectorXd zu_o;

Eigen::VectorXd zv_e;
Eigen::VectorXd zv_o;

Eigen::VectorXd lu_e;
Eigen::VectorXd lu_o;

Eigen::VectorXd lv_e;
Eigen::VectorXd lv_o;

Eigen::VectorXd rCon;

Eigen::VectorXd zCon;

Eigen::VectorXd rCon0;

Eigen::VectorXd zCon0;

Eigen::VectorXd r1_asym_e, r1_asym_o;
Eigen::VectorXd ru_asym_e, ru_asym_o;
Eigen::VectorXd rv_asym_e, rv_asym_o;
Eigen::VectorXd z1_asym_e, z1_asym_o;
Eigen::VectorXd zu_asym_e, zu_asym_o;
Eigen::VectorXd zv_asym_e, zv_asym_o;
Eigen::VectorXd lu_asym_e, lu_asym_o;
Eigen::VectorXd lv_asym_e, lv_asym_o;
Eigen::VectorXd rCon_asym;
Eigen::VectorXd zCon_asym;

Eigen::VectorXd ruFull;

Eigen::VectorXd zuFull;

Eigen::VectorXd r12;

Eigen::VectorXd ru12;

Eigen::VectorXd zu12;

Eigen::VectorXd rs;

Eigen::VectorXd zs;

Eigen::VectorXd tau;

Eigen::VectorXd gsqrt;

Eigen::VectorXd guu;
Eigen::VectorXd guv;
Eigen::VectorXd gvv;

Eigen::VectorXd bsupu;
Eigen::VectorXd bsupv;

Eigen::VectorXd bsubu;
Eigen::VectorXd bsubv;

Eigen::VectorXd totalPressure;

Eigen::VectorXd rBSq;

Eigen::VectorXd insideTotalPressure;

Eigen::VectorXd delBSq;

Eigen::VectorXd armn_e;
Eigen::VectorXd armn_o;
Eigen::VectorXd brmn_e;
Eigen::VectorXd brmn_o;
Eigen::VectorXd crmn_e;
Eigen::VectorXd crmn_o;

Eigen::VectorXd azmn_e;
Eigen::VectorXd azmn_o;
Eigen::VectorXd bzmn_e;
Eigen::VectorXd bzmn_o;
Eigen::VectorXd czmn_e;
Eigen::VectorXd czmn_o;

Eigen::VectorXd blmn_e;
Eigen::VectorXd blmn_o;
Eigen::VectorXd clmn_e;
Eigen::VectorXd clmn_o;

Eigen::VectorXd armn_asym_e, armn_asym_o;
Eigen::VectorXd brmn_asym_e, brmn_asym_o;
Eigen::VectorXd crmn_asym_e, crmn_asym_o;
Eigen::VectorXd azmn_asym_e, azmn_asym_o;
Eigen::VectorXd bzmn_asym_e, bzmn_asym_o;
Eigen::VectorXd czmn_asym_e, czmn_asym_o;
Eigen::VectorXd blmn_asym_e, blmn_asym_o;
Eigen::VectorXd clmn_asym_e, clmn_asym_o;

Eigen::VectorXd bLambda;
Eigen::VectorXd dLambda;
Eigen::VectorXd cLambda;
Eigen::VectorXd lambdaPreconditioner;

Eigen::VectorXd ax;
Eigen::VectorXd bx;
Eigen::VectorXd cx;

Eigen::VectorXd arm;
Eigen::VectorXd ard;
Eigen::VectorXd brm;
Eigen::VectorXd brd;
Eigen::VectorXd azm;
Eigen::VectorXd azd;
Eigen::VectorXd bzm;
Eigen::VectorXd bzd;

Eigen::VectorXd cxd;

bool lforbal = false;
Eigen::VectorXd cos01;
Eigen::VectorXd sin01;
Eigen::VectorXd rzu_fac;
Eigen::VectorXd rru_fac;
Eigen::VectorXd frcc_fac;
Eigen::VectorXd fzsc_fac;

Eigen::VectorXd ar;
Eigen::VectorXd dr;
Eigen::VectorXd br;
Eigen::VectorXd az;
Eigen::VectorXd dz;
Eigen::VectorXd bz;

Eigen::VectorXd xmpq;
Eigen::VectorXd faccon;

Eigen::VectorXd tcon;

Eigen::VectorXd gConEff;

Eigen::VectorXd gsc;
Eigen::VectorXd gcs;
Eigen::VectorXd gcc;
Eigen::VectorXd gss;
Eigen::VectorXd gConAsym;
Eigen::VectorXd refl;

Eigen::VectorXd gCon;

Eigen::VectorXd frcon_e;
Eigen::VectorXd frcon_o;
Eigen::VectorXd fzcon_e;
Eigen::VectorXd fzcon_o;

Eigen::VectorXd frcon_asym_e, frcon_asym_o;
Eigen::VectorXd fzcon_asym_e, fzcon_asym_o;

private:
FlowControl& m_fc_;
const Sizes& s_;
const FourierBasisFastPoloidal& t_;
RadialProfiles& m_p_;
const VmecConstants& constants_;
ThreadLocalStorage& m_ls_;
HandoverStorage& m_h_;
const RadialPartitioning& r_;

const std::vector<std::unique_ptr<FreeBoundaryBase>>* m_fb_vac_;
int m_vac_num_threads_;
VacuumPressureState& m_vacuum_pressure_state_;
std::int64_t force_evaluation_count_ = 0;

int signOfJacobian;

static constexpr double dSHalfDsInterp = 0.25;

int ncurr;

double adiabaticIndex;

double tcon0;

Eigen::VectorXi jMin;

int nvacskip;

int ivacskip;
};

}

#endif

namespace vmecpp {

struct VmecInternalResults {
int sign_of_jacobian;

double lamscale;

int num_full;

int num_half;

int nZnT_reduced;

Eigen::VectorXd sqrtSH;
Eigen::VectorXd sqrtSF;

Eigen::VectorXd sm;
Eigen::VectorXd sp;

Eigen::VectorXd phipF;

Eigen::VectorXd chipF;

Eigen::VectorXd phipH;

Eigen::VectorXd chipH;

Eigen::VectorXd currH;

Eigen::VectorXd phiF;

Eigen::VectorXd iotaF;

Eigen::VectorXd spectral_width;

Eigen::VectorXd bvcoH;

Eigen::VectorXd dVdsH;

Eigen::VectorXd massH;

Eigen::VectorXd presH;

Eigen::VectorXd iotaH;

RowMatrixXd rmncc;
RowMatrixXd rmnss;
RowMatrixXd rmnsc;
RowMatrixXd rmncs;

RowMatrixXd zmnsc;
RowMatrixXd zmncs;
RowMatrixXd zmncc;
RowMatrixXd zmnss;

RowMatrixXd lmnsc;
RowMatrixXd lmncs;
RowMatrixXd lmncc;
RowMatrixXd lmnss;

RowMatrixXd r_e;
RowMatrixXd r_o;
RowMatrixXd z_e;
RowMatrixXd z_o;

RowMatrixXd ru_e;
RowMatrixXd ru_o;
RowMatrixXd zu_e;
RowMatrixXd zu_o;

RowMatrixXd rv_e;
RowMatrixXd rv_o;
RowMatrixXd zv_e;
RowMatrixXd zv_o;

RowMatrixXd ruFull;
RowMatrixXd zuFull;

RowMatrixXd r12;

RowMatrixXd ru12;
RowMatrixXd zu12;

RowMatrixXd rs;
RowMatrixXd zs;

RowMatrixXd gsqrt;

RowMatrixXd guu;
RowMatrixXd guv;
RowMatrixXd gvv;

RowMatrixXd bsupu;
RowMatrixXd bsupv;

RowMatrixXd bsubu;
RowMatrixXd bsubv;

RowMatrixXd bsubvF;

RowMatrixXd total_pressure;

double currv;

bool operator==(const VmecInternalResults&) const = default;
bool operator!=(const VmecInternalResults& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(VmecInternalResults& m_obj,
H5::H5File& from_file);

static constexpr char H5key[] = "/vmec_internal_results";
};

struct PoloidalCurrentToFixBSubV {

Eigen::VectorXd poloidal_current_deviation;
};

struct RemainingMetric {

RowMatrixXd rv12;

RowMatrixXd zv12;

RowMatrixXd rs12;

RowMatrixXd zs12;

RowMatrixXd gsu;

RowMatrixXd gsv;

bool operator==(const RemainingMetric&) const = default;
bool operator!=(const RemainingMetric& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(RemainingMetric& m_obj, H5::H5File& from_file);

static constexpr char H5key[] = "/remaining_metric";
};

struct CylindricalComponentsOfB {

RowMatrixXd b_r;
RowMatrixXd b_phi;
RowMatrixXd b_z;

bool operator==(const CylindricalComponentsOfB&) const = default;
bool operator!=(const CylindricalComponentsOfB& o) const {
return !(*this == o);
}

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(CylindricalComponentsOfB& m_obj,
H5::H5File& from_file);

static constexpr char H5key[] = "/cylindrical_components_of_b";
};

struct BSubSHalf {

RowMatrixXd bsubs_half;

bool operator==(const BSubSHalf&) const = default;
bool operator!=(const BSubSHalf& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(BSubSHalf& m_obj, H5::H5File& from_file);

static constexpr char H5key[] = "/bsubs_half";
};

struct BSubSFull {

RowMatrixXd bsubs_full;

bool operator==(const BSubSFull&) const = default;
bool operator!=(const BSubSFull& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(BSubSFull& m_obj, H5::H5File& from_file);

static constexpr char H5key[] = "/bsubs_full";
};

struct SymmetryDecomposedCovariantB {

RowMatrixXd bsubs_s;

RowMatrixXd bsubs_a;

RowMatrixXd bsubu_s;

RowMatrixXd bsubu_a;

RowMatrixXd bsubv_s;

RowMatrixXd bsubv_a;
};

struct CovariantBDerivatives {

RowMatrixXd bsubsu;

RowMatrixXd bsubsv;

RowMatrixXd bsubuv;

RowMatrixXd bsubvu;

bool operator==(const CovariantBDerivatives&) const = default;
bool operator!=(const CovariantBDerivatives& o) const {
return !(*this == o);
}

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(CovariantBDerivatives& m_obj,
H5::H5File& from_file);

static constexpr char H5key[] = "/covariant_b_derivatives";
};

struct JxBOutFileContents {

RowMatrixXd itheta;
RowMatrixXd izeta;
RowMatrixXd bdotk;

Eigen::VectorXd amaxfor;
Eigen::VectorXd aminfor;
Eigen::VectorXd avforce;
Eigen::VectorXd pprim;
Eigen::VectorXd jdotb;
Eigen::VectorXd bdotb;
Eigen::VectorXd bdotgradv;
Eigen::VectorXd jpar2;
Eigen::VectorXd jperp2;
Eigen::VectorXd phin;

RowMatrixXd jsupu3;
RowMatrixXd jsupv3;

RowMatrixXd jsups3;

RowMatrixXd bsupu3;
RowMatrixXd bsupv3;
RowMatrixXd jcrossb;
RowMatrixXd jxb_gradp;
RowMatrixXd jdotb_sqrtg;
RowMatrixXd sqrtg3;

RowMatrixXd bsubu3;
RowMatrixXd bsubv3;

RowMatrixXd bsubs3;

bool operator==(const JxBOutFileContents&) const = default;
bool operator!=(const JxBOutFileContents& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(JxBOutFileContents& m_obj,
H5::H5File& from_file);

static constexpr char H5key[] = "/jxbout";
};

struct MercierStabilityIntermediateQuantities {

Eigen::VectorXd s;

Eigen::VectorXd shear;

Eigen::VectorXd vpp;

Eigen::VectorXd d_pressure_d_s;

Eigen::VectorXd d_toroidal_current_d_s;

Eigen::VectorXd phip_realH;
Eigen::VectorXd phip_realF;

Eigen::VectorXd vp_real;

Eigen::VectorXd torcur;

RowMatrixXd gsqrt_full;

RowMatrixXd bdotj;

RowMatrixXd gpp;

RowMatrixXd b2;

Eigen::VectorXd tpp;

Eigen::VectorXd tbb;

Eigen::VectorXd tjb;

Eigen::VectorXd tjj;

bool operator==(const MercierStabilityIntermediateQuantities&) const =
default;
bool operator!=(const MercierStabilityIntermediateQuantities& o) const {
return !(*this == o);
}

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(MercierStabilityIntermediateQuantities& m_obj,
H5::H5File& from_file);

static constexpr char H5key[] = "/mercier_intermediate";
};

struct MercierFileContents {

Eigen::VectorXd s;

Eigen::VectorXd toroidal_flux;

Eigen::VectorXd iota;

Eigen::VectorXd shear;

Eigen::VectorXd d_volume_d_s;

Eigen::VectorXd well;

Eigen::VectorXd toroidal_current;

Eigen::VectorXd d_toroidal_current_d_s;

Eigen::VectorXd pressure;

Eigen::VectorXd d_pressure_d_s;

Eigen::VectorXd DMerc;

Eigen::VectorXd Dshear;

Eigen::VectorXd Dwell;

Eigen::VectorXd Dcurr;

Eigen::VectorXd Dgeod;

bool operator==(const MercierFileContents&) const = default;
bool operator!=(const MercierFileContents& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(MercierFileContents& m_obj,
H5::H5File& from_file);

static constexpr char H5key[] = "/mercier";
};

struct Threed1FirstTableIntermediate {

RowMatrixXd tau;

Eigen::VectorXd beta_vol;

Eigen::VectorXd overr;

double beta_axis;

Eigen::VectorXd presf;

Eigen::VectorXd phipf_loc;

Eigen::VectorXd phi1;

Eigen::VectorXd chi1;

Eigen::VectorXd chi;

Eigen::VectorXd bvcoH;
Eigen::VectorXd bucoH;

Eigen::VectorXd jcurv;

Eigen::VectorXd jcuru;

Eigen::VectorXd presgrad;

Eigen::VectorXd vpphi;

Eigen::VectorXd equif;

Eigen::VectorXd bucof;

Eigen::VectorXd bvcof;

bool operator==(const Threed1FirstTableIntermediate&) const = default;
bool operator!=(const Threed1FirstTableIntermediate& o) const {
return !(*this == o);
}

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(Threed1FirstTableIntermediate& m_obj,
H5::H5File& from_file);

static constexpr char H5key[] = "/threed1_first_table_intermediate";
};

struct Threed1FirstTable {

Eigen::VectorXd s;

Eigen::VectorXd radial_force;

Eigen::VectorXd toroidal_flux;

Eigen::VectorXd iota;

Eigen::VectorXd avg_jsupu;

Eigen::VectorXd avg_jsupv;

Eigen::VectorXd d_volume_d_phi;

Eigen::VectorXd d_pressure_d_phi;

Eigen::VectorXd spectral_width;

Eigen::VectorXd pressure;

Eigen::VectorXd buco_full;

Eigen::VectorXd bvco_full;

Eigen::VectorXd j_dot_b;

Eigen::VectorXd b_dot_b;

bool operator==(const Threed1FirstTable&) const = default;
bool operator!=(const Threed1FirstTable& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(Threed1FirstTable& m_obj, H5::H5File& from_file);

static constexpr char H5key[] = "/threed1_first_table";
};

struct Threed1GeometricAndMagneticQuantitiesIntermediate {
double anorm;
double vnorm;

Eigen::VectorXd surf_area;
double circumference_sum;

double rcenin;
double aminr2in;
double bminz2in;
double bminz2;

double sump;

Eigen::VectorXd btor_vac;
Eigen::VectorXd btor1;
Eigen::VectorXd dbtor;
Eigen::VectorXd phat;
Eigen::VectorXd redge;

double delphid_exact;
double musubi;
double rshaf1;
double rshaf2;
double rshaf;

double fpsi0;

double sumbtot;
double sumbtor;
double sumbpol;
double sump20;
double sump2;

Eigen::VectorXd jPS2;
double jpar_perp_sum;
double jparPS_perp_sum;
double s2;

double fac;
Eigen::VectorXd r3v;

bool operator==(
const Threed1GeometricAndMagneticQuantitiesIntermediate&) const = default;
bool operator!=(
const Threed1GeometricAndMagneticQuantitiesIntermediate& o) const {
return !(*this == o);
}

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(
Threed1GeometricAndMagneticQuantitiesIntermediate& obj,
H5::H5File& from_file);

static constexpr char H5key[] =
"/threed1_geometric_and_magnetic_quantities_intermediate";
};

struct Threed1GeometricAndMagneticQuantities {
double toroidal_flux;

double circum_p;
double surf_area_p;

double cross_area_p;
double volume_p;

double Rmajor_p;
double Aminor_p;
double aspect;

double kappa_p;
double rcen;

double aminr1;

double pavg;
double factor;

double b0;

double rmax_surf;
double rmin_surf;
double zmax_surf;

RowMatrixXd bmin;

RowMatrixXd bmax;

Eigen::VectorXd waist;
Eigen::VectorXd height;

double betapol;
double betatot;
double betator;
double VolAvgB;
double IonLarmor;

double jpar_perp;
double jparPS_perp;

double toroidal_current;

double rbtor;
double rbtor0;

Eigen::VectorXd psi;

Eigen::VectorXd ygeo;

Eigen::VectorXd yinden;

Eigen::VectorXd yellip;

Eigen::VectorXd ytrian;

Eigen::VectorXd yshift;

Eigen::VectorXd loc_jpar_perp;
Eigen::VectorXd loc_jparPS_perp;

bool operator==(const Threed1GeometricAndMagneticQuantities&) const = default;
bool operator!=(const Threed1GeometricAndMagneticQuantities& o) const {
return !(*this == o);
}

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(Threed1GeometricAndMagneticQuantities& m_obj,
H5::H5File& from_file);

static constexpr char H5key[] = "/threed1_geometric_and_magnetic_quantities";
};

struct Threed1Volumetrics {
double int_p;
double avg_p;

double int_bpol;
double avg_bpol;

double int_btor;
double avg_btor;

double int_modb;
double avg_modb;

double int_ekin;
double avg_ekin;

bool operator==(const Threed1Volumetrics&) const = default;
bool operator!=(const Threed1Volumetrics& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(Threed1Volumetrics& m_obj,
H5::H5File& from_file);

static constexpr char H5key[] = "/threed1_volumetrics";
};

struct Threed1AxisGeometry {

Eigen::VectorXd raxis_symm;

Eigen::VectorXd zaxis_symm;

Eigen::VectorXd raxis_asym;

Eigen::VectorXd zaxis_asym;

bool operator==(const Threed1AxisGeometry&) const = default;
bool operator!=(const Threed1AxisGeometry& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(Threed1AxisGeometry& m_obj,
H5::H5File& from_file);

static constexpr char H5key[] = "/threed1_axis_geometry";
};

struct Threed1FreeBoundary {
RowMatrixXd rb;
RowMatrixXd phib;
RowMatrixXd zb;
RowMatrixXd bsqmhdi;
RowMatrixXd bsqvaci;
RowMatrixXd bsqmhdf;
RowMatrixXd bsqvacf;
RowMatrixXd bredge;
RowMatrixXd bpedge;
RowMatrixXd bzedge;
RowMatrixXd brv;
RowMatrixXd bphiv;
RowMatrixXd bzv;

bool operator==(const Threed1FreeBoundary&) const = default;
bool operator!=(const Threed1FreeBoundary& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(Threed1FreeBoundary& m_obj,
H5::H5File& from_file);

static constexpr const char* H5key = "/threed1_free_boundary";
};

struct Threed1Betas {

double betatot;

double betapol;

double betator;

double rbtor;

double betaxis;

double betstr;

bool operator==(const Threed1Betas&) const = default;
bool operator!=(const Threed1Betas& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(Threed1Betas& m_obj, H5::H5File& from_file);

static constexpr char H5key[] = "/threed1_betas";
};

struct Threed1ShafranovIntegrals {
double scaling_ratio;

double r_lao;
double f_lao;
double f_geo;

double smaleli;
double betai;
double musubi;
double lambda;

double s11;
double s12;
double s13;
double s2;
double s3;

double delta1;
double delta2;
double delta3;

bool operator==(const Threed1ShafranovIntegrals&) const = default;
bool operator!=(const Threed1ShafranovIntegrals& o) const {
return !(*this == o);
}

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(Threed1ShafranovIntegrals& m_obj,
H5::H5File& from_file);

static constexpr char H5key[] = "/threed1_shafranov_integrals";
};

struct WOutFileContents {

double version_;

std::string input_extension;

int signgs;

double gamma;

std::string pcurr_type;

std::string pmass_type;

std::string piota_type;

Eigen::VectorXd am;

Eigen::VectorXd ac;

Eigen::VectorXd ai;

Eigen::VectorXd am_aux_s;

Eigen::VectorXd am_aux_f;

Eigen::VectorXd ac_aux_s;

Eigen::VectorXd ac_aux_f;

Eigen::VectorXd ai_aux_s;

Eigen::VectorXd ai_aux_f;

int nfp;

int mpol;

int ntor;

bool lasym = false;

bool lrfp = false;

int ns;

double ftolv;

int niter;

bool lfreeb = false;

std::string mgrid_file;

int nextcur;

Eigen::VectorXd extcur;

std::string mgrid_mode;

double wb;

double wp;

double rmax_surf;

double rmin_surf;

double zmax_surf;

int mnmax;

int mnmax_nyq;

int ier_flag;

double aspect;

double betatotal;
double betapol;
double betator;
double betaxis;

double b0;

double rbtor0;
double rbtor;

double IonLarmor;
double volavgB;

double ctor;

double Aminor_p;
double Rmajor_p;
double volume;

double fsqr;
double fsqz;
double fsql;

int itfsq;

Eigen::VectorXd iotaf;

Eigen::VectorXd q_factor;

Eigen::VectorXd presf;

Eigen::VectorXd phi;

Eigen::VectorXd phipf;

Eigen::VectorXd chi;

Eigen::VectorXd chipf;

Eigen::VectorXd jcuru;
Eigen::VectorXd jcurv;

Eigen::VectorXd fsqt;
Eigen::VectorXd force_residual_r;
Eigen::VectorXd force_residual_z;
Eigen::VectorXd force_residual_lambda;
Eigen::VectorXd delbsq;
Eigen::VectorXi restart_reason_timetrace;

Eigen::VectorXd wdot;

Eigen::VectorXd iotas;
Eigen::VectorXd mass;
Eigen::VectorXd pres;
Eigen::VectorXd beta_vol;
Eigen::VectorXd buco;
Eigen::VectorXd bvco;
Eigen::VectorXd vp;
Eigen::VectorXd specw;
Eigen::VectorXd phips;
Eigen::VectorXd over_r;

Eigen::VectorXd jdotb;
Eigen::VectorXd bdotb;
Eigen::VectorXd bdotgradv;

Eigen::VectorXd DMerc;
Eigen::VectorXd DShear;
Eigen::VectorXd DWell;
Eigen::VectorXd DCurr;
Eigen::VectorXd DGeod;

Eigen::VectorXd equif;

std::vector<std::string> curlabel;

Eigen::VectorXd potvac;

Eigen::VectorXi xmpot;
Eigen::VectorXi xnpot;

Eigen::VectorXi xm;
Eigen::VectorXi xn;
Eigen::VectorXi xm_nyq;
Eigen::VectorXi xn_nyq;

Eigen::VectorXd raxis_cc;
Eigen::VectorXd zaxis_cs;

RowMatrixXd rmnc;

RowMatrixXd zmns;

RowMatrixXd lmns_full;

RowMatrixXd lmns;

RowMatrixXd gmnc;

RowMatrixXd bmnc;

RowMatrixXd bsubumnc;

RowMatrixXd bsubvmnc;

RowMatrixXd bsubsmns;

RowMatrixXd bsubsmns_full;

RowMatrixXd bsupumnc;

RowMatrixXd bsupvmnc;

RowMatrixXd currumnc;

RowMatrixXd currvmnc;

Eigen::VectorXd raxis_cs;
Eigen::VectorXd zaxis_cc;

RowMatrixXd rmns;

RowMatrixXd zmnc;

RowMatrixXd lmnc_full;

RowMatrixXd lmnc;

RowMatrixXd gmns;

RowMatrixXd bmns;

RowMatrixXd bsubumns;

RowMatrixXd bsubvmns;

RowMatrixXd bsubsmnc;

RowMatrixXd bsubsmnc_full;

RowMatrixXd bsupumns;

RowMatrixXd bsupvmns;

RowMatrixXd currumns;

RowMatrixXd currvmns;

bool operator==(const WOutFileContents&) const = default;
bool operator!=(const WOutFileContents& o) const { return !(*this == o); }

absl::Status WriteTo(H5::H5File& file) const;

static absl::Status LoadInto(WOutFileContents& m_obj, H5::H5File& from_file);

static constexpr char H5key[] = "/wout";
};

struct OutputQuantities {
VmecInternalResults vmec_internal_results;
RemainingMetric remaining_metric;
CylindricalComponentsOfB b_cylindrical;
BSubSHalf bsubs_half;
BSubSFull bsubs_full;
CovariantBDerivatives covariant_b_derivatives;
JxBOutFileContents jxbout;
MercierStabilityIntermediateQuantities mercier_intermediate;
MercierFileContents mercier;
Threed1FirstTableIntermediate threed1_first_table_intermediate;
Threed1FirstTable threed1_first_table;
Threed1GeometricAndMagneticQuantitiesIntermediate
threed1_geometric_magnetic_intermediate;
Threed1GeometricAndMagneticQuantities threed1_geometric_magnetic;
Threed1Volumetrics threed1_volumetrics;
Threed1AxisGeometry threed1_axis;
Threed1Betas threed1_betas;
Threed1FreeBoundary threed1_free_boundary;
Threed1ShafranovIntegrals threed1_shafranov_integrals;
WOutFileContents wout;
VmecINDATA indata;

bool operator==(const OutputQuantities&) const = default;
bool operator!=(const OutputQuantities& o) const { return !(*this == o); }

absl::Status Save(const std::filesystem::path& path) const;

static absl::StatusOr<OutputQuantities> Load(
const std::filesystem::path& path);

private:

absl::Status WriteTo(H5::H5File& file) const;
static absl::StatusOr<OutputQuantities> ReadFrom(H5::H5File& file);
};

Threed1FreeBoundary ComputeThreed1FreeBoundary(
const Sizes& s, const FlowControl& fc,
const HandoverStorage& handover_storage,
const VmecInternalResults& vmec_internal_results,
const CylindricalComponentsOfB& b_cylindrical);

OutputQuantities ComputeOutputQuantities(
int sign_of_jacobian, const VmecINDATA& indata, const Sizes& s,
const FlowControl& fc, const VmecConstants& constants,
const FourierBasisFastPoloidal& t, const HandoverStorage& h,
const std::string& mgrid_mode,
const std::vector<std::string>& coil_group_names,
const std::vector<std::unique_ptr<RadialPartitioning> >&
radial_partitioning,
const std::vector<std::unique_ptr<FourierGeometry> >& decomposed_x,
const std::vector<std::unique_ptr<IdealMhdModel> >& models_from_threads,
const std::vector<std::unique_ptr<RadialProfiles> >& radial_profiles,
const VmecCheckpoint& checkpoint, VacuumPressureState vacuum_pressure_state,
VmecStatus vmec_status, int iter2);

VmecInternalResults GatherDataFromThreads(
int sign_of_jacobian, const Sizes& s, const FlowControl& fc,
const VmecConstants& constants,
const std::vector<std::unique_ptr<RadialPartitioning> >&
radial_partitioning,
const std::vector<std::unique_ptr<FourierGeometry> >& decomposed_x,
const std::vector<std::unique_ptr<IdealMhdModel> >& models_from_threads,
const std::vector<std::unique_ptr<RadialProfiles> >& radial_profiles);

void MeshBledingBSubZeta(const Sizes& s, const FlowControl& fc,
VmecInternalResults& m_vmec_internal_results);

PoloidalCurrentToFixBSubV ComputePoloidalCurrentToFixBSubV(
const Sizes& s, const VmecInternalResults& vmec_internal_results);

void FixupPoloidalCurrent(
const Sizes& s,
const PoloidalCurrentToFixBSubV& poloidal_current_to_fix_bsubv,
VmecInternalResults& m_vmec_internal_results);

void RecomputeToroidalFlux(const FlowControl& fc,
VmecInternalResults& m_vmec_internal_results);

RemainingMetric ComputeRemainingMetric(
const Sizes& s, const VmecInternalResults& vmec_internal_results);

CylindricalComponentsOfB BCylindricalComponents(
const Sizes& s, const VmecInternalResults& vmec_internal_results,
const RemainingMetric& remaining_metric);

BSubSHalf ComputeBSubSOnHalfGrid(
const Sizes& s, const VmecInternalResults& vmec_internal_results,
const RemainingMetric& remaining_metric);

BSubSFull PutBSubSOnFullGrid(const Sizes& s,
const VmecInternalResults& vmec_internal_results,
const BSubSHalf& bsubs_half);

SymmetryDecomposedCovariantB DecomposeCovariantBBySymmetry(
const Sizes& s, const VmecInternalResults& vmec_internal_results,
const BSubSFull& bsubs_full);

CovariantBDerivatives LowPassFilterCovariantB(
const Sizes& s, const FourierBasisFastPoloidal& t,
const SymmetryDecomposedCovariantB& decomposed_bcov,
VmecInternalResults& m_vmec_internal_results);

void ExtrapolateBSubS(const Sizes& s, const FlowControl& fc,
BSubSFull& m_bsubs_full);

JxBOutFileContents ComputeJxBOutputFileContents(
const Sizes& s, const FlowControl& fc,
const VmecInternalResults& vmec_internal_results,
const BSubSFull& bsubs_full,
const CovariantBDerivatives& covariant_b_derivatives,
const bool return_outputs_even_if_not_converged, VmecStatus vmec_status);

MercierStabilityIntermediateQuantities ComputeIntermediateMercierQuantities(
const Sizes& s, const FlowControl& fc,
const VmecInternalResults& vmec_internal_results,
const JxBOutFileContents& jxbout);

MercierFileContents ComputeMercierStability(
const FlowControl& fc, const VmecInternalResults& vmec_internal_results,
const MercierStabilityIntermediateQuantities& mercier_intermediate);

Threed1FirstTableIntermediate ComputeIntermediateThreed1FirstTableQuantities(
const Sizes& s, const FlowControl& fc,
const VmecInternalResults& vmec_internal_results);

Threed1FirstTable ComputeThreed1FirstTable(
const FlowControl& fc, const VmecInternalResults& vmec_internal_results,
const JxBOutFileContents& jxbout,
const Threed1FirstTableIntermediate& threed1_first_table_intermediate);

Threed1GeometricAndMagneticQuantitiesIntermediate
ComputeIntermediateThreed1GeometricMagneticQuantities(
const Sizes& s, const FlowControl& fc,
const HandoverStorage& handover_storage,
const VmecInternalResults& vmec_internal_results,
const JxBOutFileContents& jxbout,
const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
VacuumPressureState vacuum_pressure_state);

Threed1GeometricAndMagneticQuantities ComputeThreed1GeometricMagneticQuantities(
const Sizes& s, const FlowControl& fc,
const HandoverStorage& handover_storage,
const VmecInternalResults& vmec_internal_results,
const JxBOutFileContents& jxbout,
const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
const Threed1GeometricAndMagneticQuantitiesIntermediate&
threed1_geometric_magnetic_intermediate);

Threed1Volumetrics ComputeThreed1Volumetrics(
const Threed1GeometricAndMagneticQuantitiesIntermediate&
threed1_geometric_magnetic_intermediate,
const Threed1GeometricAndMagneticQuantities& threed1_geomag);

Threed1AxisGeometry ComputeThreed1AxisGeometry(
const Sizes& s, const FourierBasisFastPoloidal& fourier_basis,
const VmecInternalResults& vmec_internal_results);

Threed1Betas ComputeThreed1Betas(
const HandoverStorage& handover_storage,
const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
const Threed1GeometricAndMagneticQuantitiesIntermediate&
threed1_geomag_intermediate,
const Threed1GeometricAndMagneticQuantities& threed1_geomag);

Threed1ShafranovIntegrals ComputeThreed1ShafranovIntegrals(
const Sizes& s, const FlowControl& fc,
const HandoverStorage& handover_storage,
const VmecInternalResults& vmec_internal_results,
const Threed1GeometricAndMagneticQuantitiesIntermediate&
threed1_geometric_magnetic_intermediate,
const Threed1GeometricAndMagneticQuantities& threed1_geomag,
VacuumPressureState vacuum_pressure_state);

WOutFileContents ComputeWOutFileContents(
const VmecINDATA& indata, const Sizes& s, const FourierBasisFastPoloidal& t,
const FlowControl& fc, const VmecConstants& constants,
const HandoverStorage& handover_storage, const std::string& mgrid_mode,
const std::vector<std::string>& coil_group_names,
VmecInternalResults& m_vmec_internal_results, const BSubSHalf& bsubs_half,
const BSubSFull& bsubs_full, const MercierFileContents& mercier,
const JxBOutFileContents& jxbout,
const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
const Threed1FirstTable& threed1_first_table,
const Threed1GeometricAndMagneticQuantities& threed1_geomag,
const Threed1AxisGeometry& threed1_axis, const Threed1Betas& threed1_betas,
const Threed1FreeBoundary& threed1_free_boundary, VmecStatus vmec_status,
int iter2);

}

#endif

namespace vmecpp {

enum class GeometryCoefficientState : std::uint8_t {
kSolver,
kPhysical,
};

Geometry MakeGeometry(
const VmecINDATA& indata, const VmecInternalResults& internal,
GeometryCoefficientState state = GeometryCoefficientState::kSolver);

}

#endif

#include <algorithm>
#include <numbers>
#include <vector>

namespace vmecpp {
namespace {

std::vector<double> Scale(const RowMatrixXd& source, int mpol, int ntor) {
std::vector<double> result(source.size());
for (int j = 0; j < source.rows(); ++j) {
for (int m = 0; m < mpol; ++m) {
for (int n = 0; n <= ntor; ++n) {
const int index = (j * mpol + m) * (ntor + 1) + n;
const double mscale = m == 0 ? 1.0 : std::numbers::sqrt2;
const double nscale = n == 0 ? 1.0 : std::numbers::sqrt2;
result[index] = source(j, n * mpol + m) * mscale * nscale;
}
}
}
return result;
}

void ScaleLambda(std::vector<double>& m_coefficients,
const VmecInternalResults& internal, int modes_per_surface) {
if (m_coefficients.empty()) return;
for (int j = 0; j < internal.num_full; ++j) {
const double factor = internal.lamscale / internal.phipF[j];
for (int mode = 0; mode < modes_per_surface; ++mode) {
m_coefficients[j * modes_per_surface + mode] *= factor;
}
}
}

void ConvertM1ToPhysical(GeometryCoefficients& m_coefficients,
const VmecINDATA& indata, int num_full,
int sign_of_jacobian) {

const double sigma = -sign_of_jacobian;
auto convert = [num_full, sigma, &indata](std::vector<double>& m_r,
std::vector<double>& m_z) {
if (m_r.empty() || m_z.empty()) return;
for (int j = 0; j < num_full; ++j) {
for (int n = 0; n <= indata.ntor; ++n) {
const int index = (j * indata.mpol + 1) * (indata.ntor + 1) + n;
const double old_r = m_r[index];
m_r[index] = old_r + sigma * m_z[index];
m_z[index] = sigma * old_r - m_z[index];
}
}
};

if (indata.mpol > 1 && indata.ntor > 0) {
convert(m_coefficients.r_ss, m_coefficients.z_cs);
}
if (indata.mpol > 1 && indata.lasym) {
convert(m_coefficients.r_sc, m_coefficients.z_cc);
}
}

}

Geometry MakeGeometry(const VmecINDATA& indata,
const VmecInternalResults& internal,
GeometryCoefficientState state) {
Geometry result{
.dimensions = {.ns = internal.num_full,
.mpol = indata.mpol,
.ntor = indata.ntor,
.nfp = indata.nfp},
.toroidal_flux = std::vector<double>(internal.num_full, 0.0),
.poloidal_flux = std::vector<double>(internal.num_full, 0.0),
.coefficients = {},
};
const double delta_s = 1.0 / (internal.num_full - 1);
const bool has_toroidal_flux =
internal.phiF.size() == internal.num_full &&
std::any_of(internal.phiF.data(),
internal.phiF.data() + internal.phiF.size(),
[](double value) { return value != 0.0; });
if (has_toroidal_flux) {
std::copy(internal.phiF.data(), internal.phiF.data() + internal.num_full,
result.toroidal_flux.begin());
} else {

for (int j = 1; j < internal.num_full; ++j) {
result.toroidal_flux[j] =
result.toroidal_flux[j - 1] + internal.sign_of_jacobian * 2.0 *
std::numbers::pi * delta_s *
internal.phipH[j - 1];
}
}
for (int j = 1; j < internal.num_full; ++j) {
result.poloidal_flux[j] = result.poloidal_flux[j - 1] +
internal.sign_of_jacobian * 2.0 *
std::numbers::pi * delta_s *
internal.phipH[j - 1] * internal.iotaH[j - 1];
}

GeometryCoefficients& coefficients = result.coefficients;
coefficients.r_cc = Scale(internal.rmncc, indata.mpol, indata.ntor);
coefficients.z_sc = Scale(internal.zmnsc, indata.mpol, indata.ntor);
coefficients.lambda_sc = Scale(internal.lmnsc, indata.mpol, indata.ntor);
if (indata.ntor > 0) {
coefficients.r_ss = Scale(internal.rmnss, indata.mpol, indata.ntor);
coefficients.z_cs = Scale(internal.zmncs, indata.mpol, indata.ntor);
coefficients.lambda_cs = Scale(internal.lmncs, indata.mpol, indata.ntor);
}
if (indata.lasym) {
coefficients.r_sc = Scale(internal.rmnsc, indata.mpol, indata.ntor);
coefficients.z_cc = Scale(internal.zmncc, indata.mpol, indata.ntor);
coefficients.lambda_cc = Scale(internal.lmncc, indata.mpol, indata.ntor);
if (indata.ntor > 0) {
coefficients.r_cs = Scale(internal.rmncs, indata.mpol, indata.ntor);
coefficients.z_ss = Scale(internal.zmnss, indata.mpol, indata.ntor);
coefficients.lambda_ss = Scale(internal.lmnss, indata.mpol, indata.ntor);
}
}

const int modes_per_surface = indata.mpol * (indata.ntor + 1);
ScaleLambda(coefficients.lambda_sc, internal, modes_per_surface);
ScaleLambda(coefficients.lambda_cs, internal, modes_per_surface);
ScaleLambda(coefficients.lambda_cc, internal, modes_per_surface);
ScaleLambda(coefficients.lambda_ss, internal, modes_per_surface);

if (state == GeometryCoefficientState::kSolver) {
ConvertM1ToPhysical(coefficients, indata, internal.num_full,
internal.sign_of_jacobian);
}

return result;
}

}

// source: vmecpp/vmec/handover_storage/handover_storage.cc

#include <iostream>

namespace vmecpp {

HandoverStorage::HandoverStorage(const Sizes* s) : s_(*s) {
plasmaVolume = 0.0;

fNormRZ = 0.0;
fNormL = 0.0;
fNorm1 = 0.0;

thermalEnergy = 0.0;
magneticEnergy = 0.0;
mhdEnergy = 0.0;

rBtor0 = 0.0;
rBtor = 0.0;
cTor = 0.0;

bSubUVac = 0.0;
bSubVVac = 0.0;

rCon_LCFS.setZero(s_.nZnT);
zCon_LCFS.setZero(s_.nZnT);

num_threads_ = 1;
num_basis_ = 0;

mnsize = s_.mnsize;

spectral_width_numerator_ = 0.0;
spectral_width_denominator_ = 0.0;

rAxis.setZero(s_.nZeta);
zAxis.setZero(s_.nZeta);

rCC_LCFS.setZero(mnsize);
rSS_LCFS.setZero(mnsize);
zSC_LCFS.setZero(mnsize);
zCS_LCFS.setZero(mnsize);
if (s_.lasym) {
rSC_LCFS.setZero(mnsize);
rCS_LCFS.setZero(mnsize);
zCC_LCFS.setZero(mnsize);
zSS_LCFS.setZero(mnsize);
}
}

void HandoverStorage::allocate(const RadialPartitioning& r, int ns) {

if (r.get_thread_id() == 0) {
num_threads_ = r.get_num_threads();
num_basis_ = s_.num_basis;

thread_reduce_slots.resize(num_threads_, 3);
thread_reduce_slots.setZero();

rmncc_i.resize(num_threads_, mnsize);
rmncc_i.setZero();
zmnsc_i.resize(num_threads_, mnsize);
zmnsc_i.setZero();
lmnsc_i.resize(num_threads_, mnsize);
lmnsc_i.setZero();

rmncc_o.resize(num_threads_, mnsize);
rmncc_o.setZero();
zmnsc_o.resize(num_threads_, mnsize);
zmnsc_o.setZero();
lmnsc_o.resize(num_threads_, mnsize);
lmnsc_o.setZero();

if (s_.lthreed) {
rmnss_i.resize(num_threads_, mnsize);
rmnss_i.setZero();
zmncs_i.resize(num_threads_, mnsize);
zmncs_i.setZero();
lmncs_i.resize(num_threads_, mnsize);
lmncs_i.setZero();

rmnss_o.resize(num_threads_, mnsize);
rmnss_o.setZero();
zmncs_o.resize(num_threads_, mnsize);
zmncs_o.setZero();
lmncs_o.resize(num_threads_, mnsize);
lmncs_o.setZero();
}

if (s_.lasym) {
rmnsc_i.resize(num_threads_, mnsize);
rmnsc_i.setZero();
zmncc_i.resize(num_threads_, mnsize);
zmncc_i.setZero();
lmncc_i.resize(num_threads_, mnsize);
lmncc_i.setZero();

rmnsc_o.resize(num_threads_, mnsize);
rmnsc_o.setZero();
zmncc_o.resize(num_threads_, mnsize);
zmncc_o.setZero();
lmncc_o.resize(num_threads_, mnsize);
lmncc_o.setZero();

if (s_.lthreed) {
rmncs_i.resize(num_threads_, mnsize);
rmncs_i.setZero();
zmnss_i.resize(num_threads_, mnsize);
zmnss_i.setZero();
lmnss_i.resize(num_threads_, mnsize);
lmnss_i.setZero();

rmncs_o.resize(num_threads_, mnsize);
rmncs_o.setZero();
zmnss_o.resize(num_threads_, mnsize);
zmnss_o.setZero();
lmnss_o.resize(num_threads_, mnsize);
lmnss_o.setZero();
}
}

all_ar.resize(mnsize, ns);
all_ar.setZero();
all_az.resize(mnsize, ns);
all_az.setZero();
all_dr.resize(mnsize, ns);
all_dr.setZero();
all_dz.resize(mnsize, ns);
all_dz.setZero();
all_br.resize(mnsize, ns);
all_br.setZero();
all_bz.resize(mnsize, ns);
all_bz.setZero();

all_cr.resize(mnsize);
all_cz.resize(mnsize);
for (int mn = 0; mn < mnsize; ++mn) {
all_cr[mn].resize(num_basis_, ns);
all_cr[mn].setZero();
all_cz[mn].resize(num_basis_, ns);
all_cz[mn].setZero();
}

handover_cR.resize(num_basis_, mnsize);
handover_cR.setZero();
handover_cZ.resize(num_basis_, mnsize);
handover_cZ.setZero();

handover_aR.setZero(mnsize);
handover_aZ.setZero(mnsize);
}

}

void HandoverStorage::ResetSpectralWidthAccumulators() {
spectral_width_numerator_ = 0.0;
spectral_width_denominator_ = 0.0;
}

void HandoverStorage::RegisterSpectralWidthContribution(
const SpectralWidthContribution& spectral_width_contribution) {
spectral_width_numerator_ += spectral_width_contribution.numerator;
spectral_width_denominator_ += spectral_width_contribution.denominator;
}

double HandoverStorage::VolumeAveragedSpectralWidth() const {
return spectral_width_numerator_ / spectral_width_denominator_;
}

void HandoverStorage::SetRadialExtent(const RadialExtent& radial_extent) {
radial_extent_ = radial_extent;
}

void HandoverStorage::SetGeometricOffset(
const GeometricOffset& geometric_offset) {
geometric_offset_ = geometric_offset;
}

RadialExtent HandoverStorage::GetRadialExtent() const {
return radial_extent_;
}

GeometricOffset HandoverStorage::GetGeometricOffset() const {
return geometric_offset_;
}

}

// source: vmecpp/vmec/ideal_mhd_model/dft_toroidal.cc

#include <algorithm>

#include "absl/algorithm/container.h"

namespace vmecpp {

void ForcesToFourier3DSymmFastPoloidal(
const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
const FourierBasisFastPoloidal& fb,
VacuumPressureState vacuum_pressure_state,
FourierForces& m_physical_forces) {

m_physical_forces.setZero();

int jMaxRZ = std::min(rp.nsMaxF, fc.ns - 1);

if (fc.lfreeb && vacuum_pressure_state >= VacuumPressureState::kInitialized) {

jMaxRZ = std::min(rp.nsMaxF, fc.ns);
}

const int jMinL = 1;

for (int jF = rp.nsMinF; jF < jMaxRZ; ++jF) {
const int mmax = jF == 0 ? 1 : s.mpol;
for (int m = 0; m < mmax; ++m) {
const bool m_even = m % 2 == 0;

const auto& armn = m_even ? d.armn_e : d.armn_o;
const auto& azmn = m_even ? d.azmn_e : d.azmn_o;
const auto& blmn = m_even ? d.blmn_e : d.blmn_o;
const auto& brmn = m_even ? d.brmn_e : d.brmn_o;
const auto& bzmn = m_even ? d.bzmn_e : d.bzmn_o;
const auto& clmn = m_even ? d.clmn_e : d.clmn_o;
const auto& crmn = m_even ? d.crmn_e : d.crmn_o;
const auto& czmn = m_even ? d.czmn_e : d.czmn_o;
const auto& frcon = m_even ? d.frcon_e : d.frcon_o;
const auto& fzcon = m_even ? d.fzcon_e : d.fzcon_o;

for (int k = 0; k < s.nZeta; ++k) {
double rmkcc = 0.0;
double rmkcc_n = 0.0;
double rmkss = 0.0;
double rmkss_n = 0.0;
double zmksc = 0.0;
double zmksc_n = 0.0;
double zmkcs = 0.0;
double zmkcs_n = 0.0;
double lmksc = 0.0;
double lmksc_n = 0.0;
double lmkcs = 0.0;
double lmkcs_n = 0.0;

const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;
const int idx_ml_base = m * s.nThetaReduced;

for (int l = 0; l < s.nThetaReduced; ++l) {
const int idx_kl = idx_kl_base + l;
const int idx_ml = idx_ml_base + l;

const double cosmui = fb.cosmui[idx_ml];
const double sinmui = fb.sinmui[idx_ml];
const double cosmumi = fb.cosmumi[idx_ml];
const double sinmumi = fb.sinmumi[idx_ml];

lmksc += blmn[idx_kl] * cosmumi;
lmkcs += blmn[idx_kl] * sinmumi;
lmkcs_n -= clmn[idx_kl] * cosmui;
lmksc_n -= clmn[idx_kl] * sinmui;

rmkcc_n -= crmn[idx_kl] * cosmui;
zmkcs_n -= czmn[idx_kl] * cosmui;

rmkss_n -= crmn[idx_kl] * sinmui;
zmksc_n -= czmn[idx_kl] * sinmui;

const double tempR = armn[idx_kl] + xmpq[m] * frcon[idx_kl];
const double tempZ = azmn[idx_kl] + xmpq[m] * fzcon[idx_kl];

rmkcc += tempR * cosmui + brmn[idx_kl] * sinmumi;
rmkss += tempR * sinmui + brmn[idx_kl] * cosmumi;
zmksc += tempZ * sinmui + bzmn[idx_kl] * cosmumi;
zmkcs += tempZ * cosmui + bzmn[idx_kl] * sinmumi;
}

for (int n = 0; n < s.ntor + 1; ++n) {
const int idx_mn = ((jF - rp.nsMinF) * s.mpol + m) * (s.ntor + 1) + n;
const int idx_kn = k * (s.nnyq2 + 1) + n;

const double cosnv = fb.cosnv[idx_kn];
const double sinnv = fb.sinnv[idx_kn];
const double cosnvn = fb.cosnvn[idx_kn];
const double sinnvn = fb.sinnvn[idx_kn];

m_physical_forces.frcc[idx_mn] += rmkcc * cosnv + rmkcc_n * sinnvn;
m_physical_forces.frss[idx_mn] += rmkss * sinnv + rmkss_n * cosnvn;
m_physical_forces.fzsc[idx_mn] += zmksc * cosnv + zmksc_n * sinnvn;
m_physical_forces.fzcs[idx_mn] += zmkcs * sinnv + zmkcs_n * cosnvn;

if (jMinL <= jF) {
m_physical_forces.flsc[idx_mn] += lmksc * cosnv + lmksc_n * sinnvn;
m_physical_forces.flcs[idx_mn] += lmkcs * sinnv + lmkcs_n * cosnvn;
}
}
}
}
}

for (int jF = jMaxRZ; jF < rp.nsMaxFIncludingLcfs; ++jF) {
for (int m = 0; m < s.mpol; ++m) {
const bool m_even = m % 2 == 0;

const auto& blmn = m_even ? d.blmn_e : d.blmn_o;
const auto& clmn = m_even ? d.clmn_e : d.clmn_o;

for (int k = 0; k < s.nZeta; ++k) {
double lmksc = 0.0;
double lmksc_n = 0.0;
double lmkcs = 0.0;
double lmkcs_n = 0.0;

const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;
const int idx_ml_base = m * s.nThetaReduced;

for (int l = 0; l < s.nThetaReduced; ++l) {
const int idx_kl = idx_kl_base + l;
const int idx_ml = idx_ml_base + l;

const double cosmui = fb.cosmui[idx_ml];
const double sinmui = fb.sinmui[idx_ml];
const double cosmumi = fb.cosmumi[idx_ml];
const double sinmumi = fb.sinmumi[idx_ml];

lmksc += blmn[idx_kl] * cosmumi;
lmkcs += blmn[idx_kl] * sinmumi;
lmkcs_n -= clmn[idx_kl] * cosmui;
lmksc_n -= clmn[idx_kl] * sinmui;
}

for (int n = 0; n < s.ntor + 1; ++n) {
const int idx_mn = ((jF - rp.nsMinF) * s.mpol + m) * (s.ntor + 1) + n;
const int idx_kn = k * (s.nnyq2 + 1) + n;

const double cosnv = fb.cosnv[idx_kn];
const double sinnv = fb.sinnv[idx_kn];
const double cosnvn = fb.cosnvn[idx_kn];
const double sinnvn = fb.sinnvn[idx_kn];

m_physical_forces.flsc[idx_mn] += lmksc * cosnv + lmksc_n * sinnvn;
m_physical_forces.flcs[idx_mn] += lmkcs * sinnv + lmkcs_n * cosnvn;
}
}
}
}
}

void FourierToReal3DSymmFastPoloidal(const FourierGeometry& physical_x,
const Eigen::VectorXd& xmpq,
const RadialPartitioning& r,
const Sizes& s, const RadialProfiles& rp,
const FourierBasisFastPoloidal& fb,
RealSpaceGeometry& m_geometry) {

absl::c_fill(m_geometry.r1_e, 0);
absl::c_fill(m_geometry.r1_o, 0);
absl::c_fill(m_geometry.ru_e, 0);
absl::c_fill(m_geometry.ru_o, 0);
absl::c_fill(m_geometry.rv_e, 0);
absl::c_fill(m_geometry.rv_o, 0);
absl::c_fill(m_geometry.z1_e, 0);
absl::c_fill(m_geometry.z1_o, 0);
absl::c_fill(m_geometry.zu_e, 0);
absl::c_fill(m_geometry.zu_o, 0);
absl::c_fill(m_geometry.zv_e, 0);
absl::c_fill(m_geometry.zv_o, 0);
absl::c_fill(m_geometry.lu_e, 0);
absl::c_fill(m_geometry.lu_o, 0);
absl::c_fill(m_geometry.lv_e, 0);
absl::c_fill(m_geometry.lv_o, 0);

absl::c_fill(m_geometry.rCon, 0);
absl::c_fill(m_geometry.zCon, 0);

const int nsMinF1 = r.nsMinF1;
const int nsMinF = r.nsMinF;
for (int jF = nsMinF1; jF < r.nsMaxF1; ++jF) {
for (int m = 0; m < s.mpol; ++m) {
const bool m_even = m % 2 == 0;
const int idx_ml_base = m * s.nThetaReduced;

const double con_factor =
m_even ? xmpq[m] : xmpq[m] * rp.sqrtSF[jF - nsMinF1];

auto& r1 = m_even ? m_geometry.r1_e : m_geometry.r1_o;
auto& ru = m_even ? m_geometry.ru_e : m_geometry.ru_o;
auto& rv = m_even ? m_geometry.rv_e : m_geometry.rv_o;
auto& z1 = m_even ? m_geometry.z1_e : m_geometry.z1_o;
auto& zu = m_even ? m_geometry.zu_e : m_geometry.zu_o;
auto& zv = m_even ? m_geometry.zv_e : m_geometry.zv_o;
auto& lu = m_even ? m_geometry.lu_e : m_geometry.lu_o;
auto& lv = m_even ? m_geometry.lv_e : m_geometry.lv_o;

int jMin = 1;
if (m == 0 || m == 1) {
jMin = 0;
}

if (jF < jMin) {
continue;
}

for (int k = 0; k < s.nZeta; ++k) {
double rmkcc = 0.0;
double rmkcc_n = 0.0;
double rmkss = 0.0;
double rmkss_n = 0.0;
double zmksc = 0.0;
double zmksc_n = 0.0;
double zmkcs = 0.0;
double zmkcs_n = 0.0;
double lmksc = 0.0;
double lmksc_n = 0.0;
double lmkcs = 0.0;
double lmkcs_n = 0.0;

for (int n = 0; n < s.ntor + 1; ++n) {

const int idx_kn = k * (s.nnyq2 + 1) + n;

double cosnv = fb.cosnv[idx_kn];
double sinnv = fb.sinnv[idx_kn];
double sinnvn = fb.sinnvn[idx_kn];
double cosnvn = fb.cosnvn[idx_kn];

int idx_mn = ((jF - nsMinF1) * s.mpol + m) * (s.ntor + 1) + n;

rmkcc += physical_x.rmncc[idx_mn] * cosnv;
rmkcc_n += physical_x.rmncc[idx_mn] * sinnvn;
rmkss += physical_x.rmnss[idx_mn] * sinnv;
rmkss_n += physical_x.rmnss[idx_mn] * cosnvn;
zmksc += physical_x.zmnsc[idx_mn] * cosnv;
zmksc_n += physical_x.zmnsc[idx_mn] * sinnvn;
zmkcs += physical_x.zmncs[idx_mn] * sinnv;
zmkcs_n += physical_x.zmncs[idx_mn] * cosnvn;
lmksc += physical_x.lmnsc[idx_mn] * cosnv;
lmksc_n += physical_x.lmnsc[idx_mn] * sinnvn;
lmkcs += physical_x.lmncs[idx_mn] * sinnv;
lmkcs_n += physical_x.lmncs[idx_mn] * cosnvn;
}

const int idx_kl_base = ((jF - nsMinF1) * s.nZeta + k) * s.nThetaEff;

for (int l = 0; l < s.nThetaReduced; ++l) {
const int idx_ml = idx_ml_base + l;

const double sinmum = fb.sinmum[idx_ml];
const double cosmum = fb.cosmum[idx_ml];

const int idx_kl = idx_kl_base + l;
ru[idx_kl] += rmkcc * sinmum + rmkss * cosmum;
zu[idx_kl] += zmksc * cosmum + zmkcs * sinmum;
lu[idx_kl] += lmksc * cosmum + lmkcs * sinmum;
}

for (int l = 0; l < s.nThetaReduced; ++l) {
const int idx_kl = idx_kl_base + l;
const int idx_ml = idx_ml_base + l;

const double cosmu = fb.cosmu[idx_ml];
const double sinmu = fb.sinmu[idx_ml];
rv[idx_kl] += rmkcc_n * cosmu + rmkss_n * sinmu;
zv[idx_kl] += zmksc_n * sinmu + zmkcs_n * cosmu;

lv[idx_kl] -= lmksc_n * sinmu + lmkcs_n * cosmu;
}

for (int l = 0; l < s.nThetaReduced; ++l) {
const int idx_ml = idx_ml_base + l;

const double cosmu = fb.cosmu[idx_ml];
const double sinmu = fb.sinmu[idx_ml];

const int idx_kl = idx_kl_base + l;

r1[idx_kl] += rmkcc * cosmu + rmkss * sinmu;
z1[idx_kl] += zmksc * sinmu + zmkcs * cosmu;
}

if (nsMinF <= jF && jF < r.nsMaxFIncludingLcfs) {
for (int l = 0; l < s.nThetaReduced; ++l) {
const int idx_ml = idx_ml_base + l;
const double cosmu = fb.cosmu[idx_ml];
const double sinmu = fb.sinmu[idx_ml];

const int idx_con = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff + l;
m_geometry.rCon[idx_con] +=
(rmkcc * cosmu + rmkss * sinmu) * con_factor;
m_geometry.zCon[idx_con] +=
(zmksc * sinmu + zmkcs * cosmu) * con_factor;
}
}
}
}
}
}

void FourierToReal3DAsymFastPoloidal(const FourierGeometry& physical_x,
const Eigen::VectorXd& xmpq,
const RadialPartitioning& r,
const Sizes& s, const RadialProfiles& rp,
const FourierBasisFastPoloidal& fb,
RealSpaceGeometry& m_geometry) {

absl::c_fill(m_geometry.r1_e, 0);
absl::c_fill(m_geometry.r1_o, 0);
absl::c_fill(m_geometry.ru_e, 0);
absl::c_fill(m_geometry.ru_o, 0);
absl::c_fill(m_geometry.rv_e, 0);
absl::c_fill(m_geometry.rv_o, 0);
absl::c_fill(m_geometry.z1_e, 0);
absl::c_fill(m_geometry.z1_o, 0);
absl::c_fill(m_geometry.zu_e, 0);
absl::c_fill(m_geometry.zu_o, 0);
absl::c_fill(m_geometry.zv_e, 0);
absl::c_fill(m_geometry.zv_o, 0);
absl::c_fill(m_geometry.lu_e, 0);
absl::c_fill(m_geometry.lu_o, 0);
absl::c_fill(m_geometry.lv_e, 0);
absl::c_fill(m_geometry.lv_o, 0);

absl::c_fill(m_geometry.rCon, 0);
absl::c_fill(m_geometry.zCon, 0);

const int nsMinF1 = r.nsMinF1;
const int nsMinF = r.nsMinF;
for (int jF = nsMinF1; jF < r.nsMaxF1; ++jF) {
for (int m = 0; m < s.mpol; ++m) {
const bool m_even = m % 2 == 0;
const int idx_ml_base = m * s.nThetaReduced;

const double con_factor =
m_even ? xmpq[m] : xmpq[m] * rp.sqrtSF[jF - nsMinF1];

auto& r1 = m_even ? m_geometry.r1_e : m_geometry.r1_o;
auto& ru = m_even ? m_geometry.ru_e : m_geometry.ru_o;
auto& rv = m_even ? m_geometry.rv_e : m_geometry.rv_o;
auto& z1 = m_even ? m_geometry.z1_e : m_geometry.z1_o;
auto& zu = m_even ? m_geometry.zu_e : m_geometry.zu_o;
auto& zv = m_even ? m_geometry.zv_e : m_geometry.zv_o;
auto& lu = m_even ? m_geometry.lu_e : m_geometry.lu_o;
auto& lv = m_even ? m_geometry.lv_e : m_geometry.lv_o;

int jMin = 1;
if (m == 0 || m == 1) {
jMin = 0;
}
if (jF < jMin) {
continue;
}

for (int k = 0; k < s.nZeta; ++k) {
const int idx_kn_base = k * (s.nnyq2 + 1);
const int idx_mn_base = ((jF - nsMinF1) * s.mpol + m) * (s.ntor + 1);

auto cosnv_seg = fb.cosnv.segment(idx_kn_base, s.ntor + 1);
auto sinnv_seg = fb.sinnv.segment(idx_kn_base, s.ntor + 1);
auto sinnvn_seg = fb.sinnvn.segment(idx_kn_base, s.ntor + 1);
auto cosnvn_seg = fb.cosnvn.segment(idx_kn_base, s.ntor + 1);

auto rmnsc_seg = Eigen::Map<const Eigen::VectorXd>(
physical_x.rmnsc.data() + idx_mn_base, s.ntor + 1);
auto rmncs_seg = Eigen::Map<const Eigen::VectorXd>(
physical_x.rmncs.data() + idx_mn_base, s.ntor + 1);
auto zmncc_seg = Eigen::Map<const Eigen::VectorXd>(
physical_x.zmncc.data() + idx_mn_base, s.ntor + 1);
auto zmnss_seg = Eigen::Map<const Eigen::VectorXd>(
physical_x.zmnss.data() + idx_mn_base, s.ntor + 1);
auto lmncc_seg = Eigen::Map<const Eigen::VectorXd>(
physical_x.lmncc.data() + idx_mn_base, s.ntor + 1);
auto lmnss_seg = Eigen::Map<const Eigen::VectorXd>(
physical_x.lmnss.data() + idx_mn_base, s.ntor + 1);

double rmksc = rmnsc_seg.dot(cosnv_seg);
double rmksc_n = rmnsc_seg.dot(sinnvn_seg);
double rmkcs = rmncs_seg.dot(sinnv_seg);
double rmkcs_n = rmncs_seg.dot(cosnvn_seg);
double zmkcc = zmncc_seg.dot(cosnv_seg);
double zmkcc_n = zmncc_seg.dot(sinnvn_seg);
double zmkss = zmnss_seg.dot(sinnv_seg);
double zmkss_n = zmnss_seg.dot(cosnvn_seg);
double lmkcc = lmncc_seg.dot(cosnv_seg);
double lmkcc_n = lmncc_seg.dot(sinnvn_seg);
double lmkss = lmnss_seg.dot(sinnv_seg);
double lmkss_n = lmnss_seg.dot(cosnvn_seg);

const int idx_kl_base = ((jF - nsMinF1) * s.nZeta + k) * s.nThetaEff;

auto sinmum_seg = fb.sinmum.segment(idx_ml_base, s.nThetaReduced);
auto cosmum_seg = fb.cosmum.segment(idx_ml_base, s.nThetaReduced);

auto ru_seg = Eigen::Map<Eigen::VectorXd>(ru.data() + idx_kl_base,
s.nThetaReduced);
auto zu_seg = Eigen::Map<Eigen::VectorXd>(zu.data() + idx_kl_base,
s.nThetaReduced);
auto lu_seg = Eigen::Map<Eigen::VectorXd>(lu.data() + idx_kl_base,
s.nThetaReduced);

ru_seg += rmksc * cosmum_seg + rmkcs * sinmum_seg;
zu_seg += zmkcc * sinmum_seg + zmkss * cosmum_seg;
lu_seg += lmkcc * sinmum_seg + lmkss * cosmum_seg;

auto cosmu_seg = fb.cosmu.segment(idx_ml_base, s.nThetaReduced);
auto sinmu_seg = fb.sinmu.segment(idx_ml_base, s.nThetaReduced);

auto rv_seg = Eigen::Map<Eigen::VectorXd>(rv.data() + idx_kl_base,
s.nThetaReduced);
auto zv_seg = Eigen::Map<Eigen::VectorXd>(zv.data() + idx_kl_base,
s.nThetaReduced);
auto lv_seg = Eigen::Map<Eigen::VectorXd>(lv.data() + idx_kl_base,
s.nThetaReduced);

rv_seg += rmksc_n * sinmu_seg + rmkcs_n * cosmu_seg;
zv_seg += zmkcc_n * cosmu_seg + zmkss_n * sinmu_seg;
lv_seg -= lmkcc_n * cosmu_seg + lmkss_n * sinmu_seg;

auto r1_seg = Eigen::Map<Eigen::VectorXd>(r1.data() + idx_kl_base,
s.nThetaReduced);
auto z1_seg = Eigen::Map<Eigen::VectorXd>(z1.data() + idx_kl_base,
s.nThetaReduced);

r1_seg += rmksc * sinmu_seg + rmkcs * cosmu_seg;
z1_seg += zmkcc * cosmu_seg + zmkss * sinmu_seg;

if (nsMinF <= jF && jF < r.nsMaxFIncludingLcfs) {
const int idx_con_base = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff;

auto rCon_seg = Eigen::Map<Eigen::VectorXd>(
m_geometry.rCon.data() + idx_con_base, s.nThetaReduced);
auto zCon_seg = Eigen::Map<Eigen::VectorXd>(
m_geometry.zCon.data() + idx_con_base, s.nThetaReduced);

rCon_seg += (rmksc * sinmu_seg + rmkcs * cosmu_seg) * con_factor;
zCon_seg += (zmkcc * cosmu_seg + zmkss * sinmu_seg) * con_factor;
}
}
}
}
}

void ForcesToFourier3DAsymFastPoloidal(
const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
const FourierBasisFastPoloidal& fb,
VacuumPressureState vacuum_pressure_state,
FourierForces& m_physical_forces) {

int jMaxRZ = std::min(rp.nsMaxF, fc.ns - 1);
if (fc.lfreeb && vacuum_pressure_state >= VacuumPressureState::kInitialized) {
jMaxRZ = std::min(rp.nsMaxF, fc.ns);
}

const int jMinL = 1;

for (int jF = rp.nsMinF; jF < jMaxRZ; ++jF) {
const int mmax = jF == 0 ? 1 : s.mpol;
for (int m = 0; m < mmax; ++m) {
const bool m_even = m % 2 == 0;

const auto& armn = m_even ? d.armn_e : d.armn_o;
const auto& azmn = m_even ? d.azmn_e : d.azmn_o;
const auto& blmn = m_even ? d.blmn_e : d.blmn_o;
const auto& brmn = m_even ? d.brmn_e : d.brmn_o;
const auto& bzmn = m_even ? d.bzmn_e : d.bzmn_o;
const auto& clmn = m_even ? d.clmn_e : d.clmn_o;
const auto& crmn = m_even ? d.crmn_e : d.crmn_o;
const auto& czmn = m_even ? d.czmn_e : d.czmn_o;
const auto& frcon = m_even ? d.frcon_e : d.frcon_o;
const auto& fzcon = m_even ? d.fzcon_e : d.fzcon_o;

for (int k = 0; k < s.nZeta; ++k) {
const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;
const int idx_ml_base = m * s.nThetaReduced;

auto cosmui_seg = fb.cosmui.segment(idx_ml_base, s.nThetaReduced);
auto sinmui_seg = fb.sinmui.segment(idx_ml_base, s.nThetaReduced);
auto cosmumi_seg = fb.cosmumi.segment(idx_ml_base, s.nThetaReduced);
auto sinmumi_seg = fb.sinmumi.segment(idx_ml_base, s.nThetaReduced);

auto blmn_seg = Eigen::Map<const Eigen::VectorXd>(
blmn.data() + idx_kl_base, s.nThetaReduced);
auto clmn_seg = Eigen::Map<const Eigen::VectorXd>(
clmn.data() + idx_kl_base, s.nThetaReduced);
auto crmn_seg = Eigen::Map<const Eigen::VectorXd>(
crmn.data() + idx_kl_base, s.nThetaReduced);
auto czmn_seg = Eigen::Map<const Eigen::VectorXd>(
czmn.data() + idx_kl_base, s.nThetaReduced);
auto armn_seg = Eigen::Map<const Eigen::VectorXd>(
armn.data() + idx_kl_base, s.nThetaReduced);
auto azmn_seg = Eigen::Map<const Eigen::VectorXd>(
azmn.data() + idx_kl_base, s.nThetaReduced);
auto brmn_seg = Eigen::Map<const Eigen::VectorXd>(
brmn.data() + idx_kl_base, s.nThetaReduced);
auto bzmn_seg = Eigen::Map<const Eigen::VectorXd>(
bzmn.data() + idx_kl_base, s.nThetaReduced);
auto frcon_seg = Eigen::Map<const Eigen::VectorXd>(
frcon.data() + idx_kl_base, s.nThetaReduced);
auto fzcon_seg = Eigen::Map<const Eigen::VectorXd>(
fzcon.data() + idx_kl_base, s.nThetaReduced);

double lmkcc = blmn_seg.dot(sinmumi_seg);
double lmkss = blmn_seg.dot(cosmumi_seg);
double lmkcc_n = -clmn_seg.dot(cosmui_seg);
double lmkss_n = -clmn_seg.dot(sinmui_seg);

double rmksc_n = -crmn_seg.dot(sinmui_seg);
double zmkcc_n = -czmn_seg.dot(cosmui_seg);
double rmkcs_n = -crmn_seg.dot(cosmui_seg);
double zmkss_n = -czmn_seg.dot(sinmui_seg);

const Eigen::VectorXd tempR_seg =
(armn_seg + xmpq[m] * frcon_seg).eval();
const Eigen::VectorXd tempZ_seg =
(azmn_seg + xmpq[m] * fzcon_seg).eval();

double rmksc = tempR_seg.dot(sinmui_seg) + brmn_seg.dot(cosmumi_seg);
double rmkcs = tempR_seg.dot(cosmui_seg) + brmn_seg.dot(sinmumi_seg);
double zmkcc = tempZ_seg.dot(cosmui_seg) + bzmn_seg.dot(sinmumi_seg);
double zmkss = tempZ_seg.dot(sinmui_seg) + bzmn_seg.dot(cosmumi_seg);

const int ntorp1 = s.ntor + 1;
const int idx_mn_base = ((jF - rp.nsMinF) * s.mpol + m) * ntorp1;
const int idx_kn_base = k * (s.nnyq2 + 1);

auto cosnv_seg = fb.cosnv.segment(idx_kn_base, ntorp1);
auto sinnv_seg = fb.sinnv.segment(idx_kn_base, ntorp1);
auto cosnvn_seg = fb.cosnvn.segment(idx_kn_base, ntorp1);
auto sinnvn_seg = fb.sinnvn.segment(idx_kn_base, ntorp1);

Eigen::Map<Eigen::VectorXd> frsc_seg(
m_physical_forces.frsc.data() + idx_mn_base, ntorp1);
Eigen::Map<Eigen::VectorXd> frcs_seg(
m_physical_forces.frcs.data() + idx_mn_base, ntorp1);
Eigen::Map<Eigen::VectorXd> fzcc_seg(
m_physical_forces.fzcc.data() + idx_mn_base, ntorp1);
Eigen::Map<Eigen::VectorXd> fzss_seg(
m_physical_forces.fzss.data() + idx_mn_base, ntorp1);

frsc_seg += rmksc * cosnv_seg + rmksc_n * sinnvn_seg;
frcs_seg += rmkcs * sinnv_seg + rmkcs_n * cosnvn_seg;
fzcc_seg += zmkcc * cosnv_seg + zmkcc_n * sinnvn_seg;
fzss_seg += zmkss * sinnv_seg + zmkss_n * cosnvn_seg;

if (jMinL <= jF) {
Eigen::Map<Eigen::VectorXd> flcc_seg(
m_physical_forces.flcc.data() + idx_mn_base, ntorp1);
Eigen::Map<Eigen::VectorXd> flss_seg(
m_physical_forces.flss.data() + idx_mn_base, ntorp1);
flcc_seg += lmkcc * cosnv_seg + lmkcc_n * sinnvn_seg;
flss_seg += lmkss * sinnv_seg + lmkss_n * cosnvn_seg;
}
}
}
}

for (int jF = jMaxRZ; jF < rp.nsMaxFIncludingLcfs; ++jF) {
for (int m = 0; m < s.mpol; ++m) {
const bool m_even = m % 2 == 0;

const auto& blmn = m_even ? d.blmn_e : d.blmn_o;
const auto& clmn = m_even ? d.clmn_e : d.clmn_o;

for (int k = 0; k < s.nZeta; ++k) {
const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;
const int idx_ml_base = m * s.nThetaReduced;

auto cosmui_seg = fb.cosmui.segment(idx_ml_base, s.nThetaReduced);
auto sinmui_seg = fb.sinmui.segment(idx_ml_base, s.nThetaReduced);
auto cosmumi_seg = fb.cosmumi.segment(idx_ml_base, s.nThetaReduced);
auto sinmumi_seg = fb.sinmumi.segment(idx_ml_base, s.nThetaReduced);

auto blmn_seg = Eigen::Map<const Eigen::VectorXd>(
blmn.data() + idx_kl_base, s.nThetaReduced);
auto clmn_seg = Eigen::Map<const Eigen::VectorXd>(
clmn.data() + idx_kl_base, s.nThetaReduced);

double lmkcc = blmn_seg.dot(sinmumi_seg);
double lmkss = blmn_seg.dot(cosmumi_seg);
double lmkcc_n = -clmn_seg.dot(cosmui_seg);
double lmkss_n = -clmn_seg.dot(sinmui_seg);

const int ntorp1 = s.ntor + 1;
const int idx_mn_base = ((jF - rp.nsMinF) * s.mpol + m) * ntorp1;
const int idx_kn_base = k * (s.nnyq2 + 1);

auto cosnv_seg = fb.cosnv.segment(idx_kn_base, ntorp1);
auto sinnv_seg = fb.sinnv.segment(idx_kn_base, ntorp1);
auto cosnvn_seg = fb.cosnvn.segment(idx_kn_base, ntorp1);
auto sinnvn_seg = fb.sinnvn.segment(idx_kn_base, ntorp1);

Eigen::Map<Eigen::VectorXd> flcc_seg(
m_physical_forces.flcc.data() + idx_mn_base, ntorp1);
Eigen::Map<Eigen::VectorXd> flss_seg(
m_physical_forces.flss.data() + idx_mn_base, ntorp1);

flcc_seg += lmkcc * cosnv_seg + lmkcc_n * sinnvn_seg;
flss_seg += lmkss * sinnv_seg + lmkss_n * cosnvn_seg;
}
}
}
}

}

// source: vmecpp/vmec/ideal_mhd_model/fft_toroidal.cc

// header: vmecpp/vmec/ideal_mhd_model/fft_toroidal.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_FFT_TOROIDAL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_FFT_TOROIDAL_H_

#endif

// source: vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <numbers>
#include <span>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"

// header: vmecpp/vmec/ideal_mhd_model/exact_force_jvp.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_EXACT_FORCE_JVP_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_EXACT_FORCE_JVP_H_

namespace vmecpp {

void ExactForceDensityJvp(const double* geom, const double* dgeom, double* work,
double* dwork, double* force, double* dforce,
const LocalForceComposition* c);

}

#endif

// header: vmecpp/vmec/ideal_mhd_model/exact_force_vjp.h

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_EXACT_FORCE_VJP_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_EXACT_FORCE_VJP_H_

namespace vmecpp {

void ExactForceDensityVjp(const double* geom, double* geom_bar, double* work,
double* work_bar, double* force, double* force_bar,
const LocalForceComposition* c);

}

#endif

using vmecpp::vmec_algorithm_constants::kEvenParity;
using vmecpp::vmec_algorithm_constants::kLambdaHighMDampingMaxPower;
using vmecpp::vmec_algorithm_constants::kLambdaHighMDampingReferenceM;
using vmecpp::vmec_algorithm_constants::kLambdaPreconditionerDampingFactor;
using vmecpp::vmec_algorithm_constants::kLambdaPreconditionerZeroGuard;
using vmecpp::vmec_algorithm_constants::kOddParity;

namespace {

void HandOverBoundaryGeometry(vmecpp::HandoverStorage& m_h,
const vmecpp::FourierGeometry& physical_x,
const vmecpp::Sizes& sizes, int offset) {
const int ntorp1 = sizes.ntor + 1;
for (int m = 0; m < sizes.mpol; ++m) {
for (int n = 0; n < ntorp1; ++n) {
const int idx_mn = m * ntorp1 + n;
const int idx_nm = n * sizes.mpol + m;
m_h.rCC_LCFS[idx_nm] = physical_x.rmncc[offset + idx_mn];
m_h.zSC_LCFS[idx_nm] = physical_x.zmnsc[offset + idx_mn];

if (sizes.lthreed) {
m_h.rSS_LCFS[idx_nm] = physical_x.rmnss[offset + idx_mn];
m_h.zCS_LCFS[idx_nm] = physical_x.zmncs[offset + idx_mn];
}

if (sizes.lasym) {
m_h.rSC_LCFS[idx_nm] = physical_x.rmnsc[offset + idx_mn];
m_h.zCC_LCFS[idx_nm] = physical_x.zmncc[offset + idx_mn];
if (sizes.lthreed) {
m_h.rCS_LCFS[idx_nm] = physical_x.rmncs[offset + idx_mn];
m_h.zSS_LCFS[idx_nm] = physical_x.zmnss[offset + idx_mn];
}
}
}
}
}

void HandOverMagneticAxis(vmecpp::HandoverStorage& m_h,
const Eigen::VectorXd& r1_e,
const Eigen::VectorXd& z1_e, const vmecpp::Sizes& s) {
for (int k = 0; k < s.nZeta; ++k) {

int idx_kl = k * s.nThetaEff;
m_h.rAxis[k] = r1_e[idx_kl];
m_h.zAxis[k] = z1_e[idx_kl];
}
}

}

void vmecpp::deAliasConstraintForce(
const vmecpp::RadialPartitioning& rp,
const vmecpp::FourierBasisFastPoloidal& fb, const vmecpp::Sizes& s_,
const Eigen::VectorXd& faccon, const Eigen::VectorXd& tcon,
const Eigen::VectorXd& gConEff, Eigen::VectorXd& m_gsc,
Eigen::VectorXd& m_gcs, Eigen::VectorXd& m_gCon) {
if (!s_.lasym) {
ComputeDeAliasConstraintForce(
gConEff.data(), faccon.data(), tcon.data(), fb.sinmui.data(),
fb.cosmui.data(), fb.cosnv.data(), fb.sinnv.data(), fb.sinmu.data(),
fb.cosmu.data(), rp.nsMinF, rp.nsMaxF, s_.nZeta, s_.nThetaEff,
s_.nThetaReduced, s_.mpol, s_.ntor, s_.nnyq2, m_gsc.data(),
m_gcs.data(), m_gCon.data());
return;
}
Eigen::VectorXd gcc;
Eigen::VectorXd gss;
Eigen::VectorXd gConAsym;
Eigen::VectorXd refl;
if (s_.lasym) {
gcc.setZero(s_.ntor + 1);
gss.setZero(s_.ntor + 1);
gConAsym.setZero(s_.nZnT);
refl.setZero(s_.nThetaReduced);
}
deAliasConstraintForce(rp, fb, s_, faccon, tcon, gConEff, m_gsc, m_gcs, gcc,
gss, gConAsym, refl, m_gCon);
}

void vmecpp::deAliasConstraintForce(
const vmecpp::RadialPartitioning& rp,
const vmecpp::FourierBasisFastPoloidal& fb, const vmecpp::Sizes& s_,
const Eigen::VectorXd& faccon, const Eigen::VectorXd& tcon,
const Eigen::VectorXd& gConEff, Eigen::VectorXd& m_gsc,
Eigen::VectorXd& m_gcs, Eigen::VectorXd& m_gcc, Eigen::VectorXd& m_gss,
Eigen::VectorXd& m_gConAsym, Eigen::VectorXd& m_refl,
Eigen::VectorXd& m_gCon) {
ComputeDeAliasConstraintForce(
gConEff.data(), faccon.data(), tcon.data(), fb.sinmui.data(),
fb.cosmui.data(), fb.cosnv.data(), fb.sinnv.data(), fb.sinmu.data(),
fb.cosmu.data(), rp.nsMinF, rp.nsMaxF, s_.nZeta, s_.nThetaEff,
s_.nThetaReduced, s_.nThetaEven, s_.mpol, s_.ntor, s_.nnyq2, s_.lasym,
m_gsc.data(), m_gcs.data(), m_gcc.data(), m_gss.data(), m_gConAsym.data(),
m_refl.data(), m_gCon.data());
}

namespace vmecpp {

IdealMhdModel::IdealMhdModel(
FlowControl* m_fc, const Sizes* s, const FourierBasisFastPoloidal* t,
RadialProfiles* m_p, const VmecConstants* constants,
ThreadLocalStorage* m_ls, HandoverStorage* m_h, const RadialPartitioning* r,
const std::vector<std::unique_ptr<FreeBoundaryBase>>* m_fb_vac,
int vac_num_threads, int signOfJacobian, int nvacskip,
VacuumPressureState* m_vacuum_pressure_state)
: m_fc_(*m_fc),
s_(*s),
t_(*t),
m_p_(*m_p),
constants_(*constants),
m_ls_(*m_ls),
m_h_(*m_h),
r_(*r),
m_fb_vac_(m_fb_vac),
m_vac_num_threads_(vac_num_threads),
m_vacuum_pressure_state_(*m_vacuum_pressure_state),
signOfJacobian(signOfJacobian),
nvacskip(nvacskip),
ivacskip(0) {
CHECK_GE(nvacskip, 0)
<< "Should never happen: should be checked by VmecINDATA";
if (m_fc_.lfreeb) {
CHECK(m_fb_vac_ != nullptr && !m_fb_vac_->empty())
<< "Free-boundary configuration requires a Free-boundary solver";
CHECK_GT(m_vac_num_threads_, 0)
<< "Free-boundary configuration requires vac_num_threads > 0";
}

ncurr = 0;
adiabaticIndex = 0.0;
tcon0 = 0.0;

xmpq.setZero(s_.mpol);
faccon.setZero(s_.mpol);
for (int m = 0; m < s_.mpol; ++m) {
xmpq[m] = m * (m - 1);
if (m > 1) {
faccon[m - 1] = -0.25 * signOfJacobian / (xmpq[m] * xmpq[m]);
}
}

int nrzt1 = s_.nZnT * (r_.nsMaxF1 - r_.nsMinF1);
int nrzt = s_.nZnT * (r_.nsMaxF - r_.nsMinF);

r1_e.setZero(nrzt1);
r1_o.setZero(nrzt1);
ru_e.setZero(nrzt1);
ru_o.setZero(nrzt1);
z1_e.setZero(nrzt1);
z1_o.setZero(nrzt1);
zu_e.setZero(nrzt1);
zu_o.setZero(nrzt1);
lu_e.setZero(nrzt1);
lu_o.setZero(nrzt1);

if (s_.lthreed) {
rv_e.setZero(nrzt1);
rv_o.setZero(nrzt1);
zv_e.setZero(nrzt1);
zv_o.setZero(nrzt1);
lv_e.setZero(nrzt1);
lv_o.setZero(nrzt1);
}

int nrztIncludingBoundary = s_.nZnT * (r_.nsMaxFIncludingLcfs - r_.nsMinF);

ruFull.setZero(nrztIncludingBoundary);
zuFull.setZero(nrztIncludingBoundary);

rCon.setZero(nrztIncludingBoundary);
zCon.setZero(nrztIncludingBoundary);

rCon0.setZero(nrztIncludingBoundary);
zCon0.setZero(nrztIncludingBoundary);

r12.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
ru12.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
zu12.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
rs.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
zs.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
tau.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);

gsqrt.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);

guu.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
if (s_.lthreed) {
guv.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
}
gvv.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);

bsupu.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
bsupv.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);

bsubu.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
bsubv.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);

totalPressure.setZero((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
rBSq.setZero(s_.nZnT);

insideTotalPressure.setZero(s_.nZnT);
delBSq.setZero(s_.nZnT);

armn_e.setZero(nrzt);
armn_o.setZero(nrzt);
brmn_e.setZero(nrzt);
brmn_o.setZero(nrzt);
azmn_e.setZero(nrzt);
azmn_o.setZero(nrzt);
bzmn_e.setZero(nrzt);
bzmn_o.setZero(nrzt);
blmn_e.setZero(nrztIncludingBoundary);
blmn_o.setZero(nrztIncludingBoundary);

if (s_.lthreed) {
crmn_e.setZero(nrzt);
crmn_o.setZero(nrzt);
czmn_e.setZero(nrzt);
czmn_o.setZero(nrzt);
clmn_e.setZero(nrztIncludingBoundary);
clmn_o.setZero(nrztIncludingBoundary);
}

bLambda.setZero(r_.nsMaxF1 - r_.nsMinF1 + 1);
dLambda.setZero(r_.nsMaxF1 - r_.nsMinF1 + 1);
cLambda.setZero(r_.nsMaxF1 - r_.nsMinF1 + 1);
lambdaPreconditioner.setZero((r_.nsMaxFIncludingLcfs - r_.nsMinF) * s_.mpol *
(s_.ntor + 1));

ax.setZero((r_.nsMaxH - r_.nsMinH) * 4);
bx.setZero((r_.nsMaxH - r_.nsMinH) * 3);
cx.setZero(r_.nsMaxH - r_.nsMinH);

arm.setZero((r_.nsMaxH - r_.nsMinH) * 2);
azm.setZero((r_.nsMaxH - r_.nsMinH) * 2);
brm.setZero((r_.nsMaxH - r_.nsMinH) * 2);
bzm.setZero((r_.nsMaxH - r_.nsMinH) * 2);

ard.setZero((r_.nsMaxF - r_.nsMinF) * 2);
brd.setZero((r_.nsMaxF - r_.nsMinF) * 2);
azd.setZero((r_.nsMaxF - r_.nsMinF) * 2);
bzd.setZero((r_.nsMaxF - r_.nsMinF) * 2);
cxd.setZero(r_.nsMaxF - r_.nsMinF);

ar.setZero((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);
az.setZero((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);
dr.setZero((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);
dz.setZero((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);
br.setZero((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);
bz.setZero((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);

tcon.setZero(r_.nsMaxFIncludingLcfs - r_.nsMinF);

gConEff.setZero(nrztIncludingBoundary);
gsc.setZero(s_.ntor + 1);
gcs.setZero(s_.ntor + 1);
if (s_.lasym) {
gcc.setZero(s_.ntor + 1);
gss.setZero(s_.ntor + 1);
gConAsym.setZero(s_.nZnT);
refl.setZero(s_.nThetaReduced);
}
gCon.setZero(nrztIncludingBoundary);

frcon_e.setZero(nrzt);
frcon_o.setZero(nrzt);
fzcon_e.setZero(nrzt);
fzcon_o.setZero(nrzt);

jMin.setZero(s_.mpol * (s_.ntor + 1));

if (s_.lasym) {
r1_asym_e.setZero(nrzt1);
r1_asym_o.setZero(nrzt1);
ru_asym_e.setZero(nrzt1);
ru_asym_o.setZero(nrzt1);
z1_asym_e.setZero(nrzt1);
z1_asym_o.setZero(nrzt1);
zu_asym_e.setZero(nrzt1);
zu_asym_o.setZero(nrzt1);
lu_asym_e.setZero(nrzt1);
lu_asym_o.setZero(nrzt1);
rCon_asym.setZero(nrztIncludingBoundary);
zCon_asym.setZero(nrztIncludingBoundary);

armn_asym_e.setZero(nrzt);
armn_asym_o.setZero(nrzt);
brmn_asym_e.setZero(nrzt);
brmn_asym_o.setZero(nrzt);
azmn_asym_e.setZero(nrzt);
azmn_asym_o.setZero(nrzt);
bzmn_asym_e.setZero(nrzt);
bzmn_asym_o.setZero(nrzt);
blmn_asym_e.setZero(nrztIncludingBoundary);
blmn_asym_o.setZero(nrztIncludingBoundary);
frcon_asym_e.setZero(nrzt);
frcon_asym_o.setZero(nrzt);
fzcon_asym_e.setZero(nrzt);
fzcon_asym_o.setZero(nrzt);

if (s_.lthreed) {
rv_asym_e.setZero(nrzt1);
rv_asym_o.setZero(nrzt1);
zv_asym_e.setZero(nrzt1);
zv_asym_o.setZero(nrzt1);
lv_asym_e.setZero(nrzt1);
lv_asym_o.setZero(nrzt1);
crmn_asym_e.setZero(nrzt);
crmn_asym_o.setZero(nrzt);
czmn_asym_e.setZero(nrzt);
czmn_asym_o.setZero(nrzt);
clmn_asym_e.setZero(nrztIncludingBoundary);
clmn_asym_o.setZero(nrztIncludingBoundary);
}
}
}

void IdealMhdModel::setFromINDATA(int ncurr, double adiabaticIndex,
double tcon0, bool lforbal) {
this->ncurr = ncurr;
this->adiabaticIndex = adiabaticIndex;
this->tcon0 = tcon0;

this->lforbal = lforbal && !s_.lasym;

if (this->lforbal) {

const int num_full = r_.nsMaxF - r_.nsMinF;
rzu_fac.setZero(num_full);
rru_fac.setZero(num_full);
frcc_fac.setZero(num_full);
fzsc_fac.setZero(num_full);
cos01.setZero(s_.nZnT);
sin01.setZero(s_.nZnT);
for (int kl = 0; kl < s_.nZnT; ++kl) {
const int l = kl % s_.nThetaEff;

cos01[kl] = t_.cosmu[s_.nThetaReduced + l];
sin01[kl] = -t_.sinmu[s_.nThetaReduced + l];
}
}
}

void IdealMhdModel::evalFResInvar(const Eigen::Vector3d& localFResInvar) {
#ifdef _OPENMP
#pragma omp single
#endif
{
m_fc_.fResInvar[0] = 0.0;
m_fc_.fResInvar[1] = 0.0;
m_fc_.fResInvar[2] = 0.0;
}

SumOverThreads(localFResInvar.data(), 3, r_.get_thread_id(),
r_.get_num_threads(), m_h_.thread_reduce_slots.data(),
m_fc_.fResInvar.data());

#ifdef _OPENMP
#pragma omp single
#endif
{

constexpr double r1scale = 0.25;

m_fc_.fsqr = m_fc_.fResInvar[0] * m_h_.fNormRZ * r1scale;
m_fc_.fsqz = m_fc_.fResInvar[1] * m_h_.fNormRZ * r1scale;
m_fc_.fsql = m_fc_.fResInvar[2] * m_h_.fNormL;
}
}

void IdealMhdModel::evalFResPrecd(const Eigen::Vector3d& localFResPrecd) {
#ifdef _OPENMP
#pragma omp single
#endif
{
m_fc_.fResPrecd[0] = 0.0;
m_fc_.fResPrecd[1] = 0.0;
m_fc_.fResPrecd[2] = 0.0;
}

SumOverThreads(localFResPrecd.data(), 3, r_.get_thread_id(),
r_.get_num_threads(), m_h_.thread_reduce_slots.data(),
m_fc_.fResPrecd.data());

#ifdef _OPENMP
#pragma omp single
#endif
{
m_fc_.fsqr1 = m_fc_.fResPrecd[0] * m_h_.fNorm1;
m_fc_.fsqz1 = m_fc_.fResPrecd[1] * m_h_.fNorm1;
m_fc_.fsql1 = m_fc_.fResPrecd[2] * m_fc_.deltaS;
}
}

absl::StatusOr<bool> IdealMhdModel::update(
FourierGeometry& m_decomposed_x, FourierGeometry& m_physical_x,
FourierForces& m_decomposed_f, FourierForces& m_physical_f,
bool& m_need_restart, int& m_last_preconditioner_update,
int& m_last_full_update_nestor, FlowControl& m_fc, const int iter1,
const int iter2, const VmecCheckpoint& checkpoint,
const int iterations_before_checkpointing, bool verbose,
bool always_fix_m1_gauge) {
++force_evaluation_count_;

if (s_.mpolGeometry < s_.mpol || s_.ntorGeometry < s_.ntor) {
m_decomposed_x.maskGeometryAbove(s_.mpolGeometry, s_.ntorGeometry);
}

m_decomposed_x.decomposeInto(m_physical_x, m_p_.scalxc);
if (checkpoint == VmecCheckpoint::FOURIER_GEOMETRY_TO_START_WITH &&
iter2 >= iterations_before_checkpointing) {
return true;
}

m_physical_x.m1Constraint(1.0, signOfJacobian);

m_physical_x.extrapolateTowardsAxis();

geometryFromFourier(m_physical_x);
if (checkpoint == VmecCheckpoint::INV_DFT_GEOMETRY &&
iter2 >= iterations_before_checkpointing) {
return true;
}

if (iter2 == iter1 &&
(m_vacuum_pressure_state_ == VacuumPressureState::kOff ||
m_vacuum_pressure_state_ == VacuumPressureState::kInitializing)) {
rzConIntoVolume();
}

computeJacobian();
if (checkpoint == VmecCheckpoint::JACOBIAN &&
iter2 >= iterations_before_checkpointing) {
return true;
}

if (m_fc_.restart_reason == RestartReason::BAD_JACOBIAN) {

return false;
}

computeMetricElements();
if (checkpoint == VmecCheckpoint::METRIC &&
iter2 >= iterations_before_checkpointing) {
return true;
}

updateDifferentialVolume();

if (iter2 == 1) {
computeInitialVolume();
}
if (checkpoint == VmecCheckpoint::VOLUME &&
iter2 >= iterations_before_checkpointing) {
return true;
}

computeBContra();
if (checkpoint == VmecCheckpoint::B_CONTRA &&
iter2 >= iterations_before_checkpointing) {
return true;
}

computeBCo();
if (checkpoint == VmecCheckpoint::B_CO &&
iter2 >= iterations_before_checkpointing) {
return true;
}

pressureAndEnergies();
if (checkpoint == VmecCheckpoint::ENERGY &&
iter2 >= iterations_before_checkpointing) {
return true;
}

radialForceBalance();
if (checkpoint == VmecCheckpoint::RADIAL_FORCE_BALANCE &&
iter2 >= iterations_before_checkpointing) {
return true;
}

if (r_.nsMinH == 0) {

m_h_.rBtor0 = 1.5 * m_p_.bvcoH[r_.nsMinH - r_.nsMinH] -
0.5 * m_p_.bvcoH[r_.nsMinH + 1 - r_.nsMinH];
}

if (r_.nsMaxH == m_fc_.ns - 1) {

m_h_.rBtor = 1.5 * m_p_.bvcoH[r_.nsMaxH - 1 - r_.nsMinH] -
0.5 * m_p_.bvcoH[r_.nsMaxH - 2 - r_.nsMinH];

m_h_.cTor = (1.5 * m_p_.bucoH[r_.nsMaxH - 1 - r_.nsMinH] -
0.5 * m_p_.bucoH[r_.nsMaxH - 2 - r_.nsMinH]) *
signOfJacobian * 2.0 * M_PI;
}

hybridLambdaForce();
if (checkpoint == VmecCheckpoint::HYBRID_LAMBDA_FORCE &&
iter2 >= iterations_before_checkpointing) {
return true;
}

if (shouldUpdateRadialPreconditioner(iter1, iter2)) {
#ifdef _OPENMP
#pragma omp single nowait
#endif
{
m_last_preconditioner_update = iter2;
}

updateRadialPreconditioner();
if (checkpoint == VmecCheckpoint::UPDATE_RADIAL_PRECONDITIONER &&
iter2 >= iterations_before_checkpointing) {
return true;
}

updateVolume();

computeForceNorms(m_decomposed_x);
if (checkpoint == VmecCheckpoint::UPDATE_FORCE_NORMS &&
iter2 >= iterations_before_checkpointing) {
return true;
}

absl::Status s = constraintForceMultiplier();
if (!s.ok()) {
return s;
}
if (checkpoint == VmecCheckpoint::UPDATE_TCON &&
iter2 >= iterations_before_checkpointing) {
return true;
}
}

if ((checkpoint == VmecCheckpoint::UPDATE_RADIAL_PRECONDITIONER ||
checkpoint == VmecCheckpoint::UPDATE_FORCE_NORMS ||
checkpoint == VmecCheckpoint::UPDATE_TCON) &&
iter2 >= iterations_before_checkpointing) {
return true;
}

if (m_fc_.lfreeb && (iter2 > 1 || m_vacuum_pressure_state_ >=
VacuumPressureState::kInitialized)) {

#ifdef _OPENMP
#pragma omp barrier
#endif

ivacskip = (iter2 - iter1) % nvacskip;

if (m_vacuum_pressure_state_ != VacuumPressureState::kSettled &&
m_fc_.fsqr + m_fc_.fsqz < 1.0e-3) {

#ifdef _OPENMP
#pragma omp barrier
#endif

#ifdef _OPENMP
#pragma omp single
#endif

m_vacuum_pressure_state_ = static_cast<VacuumPressureState>(
static_cast<int>(m_vacuum_pressure_state_) + 1);
}

if (m_vacuum_pressure_state_ <= VacuumPressureState::kActive) {
ivacskip = 0;
}

if (ivacskip == 0) {
const int new_nvacskip = static_cast<int>(
1.0 / std::max(0.1, 1.0e11 * (m_fc_.fsqr + m_fc_.fsqz)));
nvacskip = std::max(nvacskip, new_nvacskip);

#ifdef _OPENMP
#pragma omp single nowait
#endif
{
m_last_full_update_nestor = iter2;
}
}

#ifdef _OPENMP
#pragma omp barrier
#endif
if (m_vacuum_pressure_state_ != VacuumPressureState::kOff) {

for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
for (int kl = 0; kl < s_.nZnT; ++kl) {
int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;

rCon0[idx_kl] *= 0.9;
zCon0[idx_kl] *= 0.9;
}
}

if (r_.nsMaxF1 == m_fc_.ns) {

HandOverBoundaryGeometry(
m_h_, m_physical_x, s_,
(r_.nsMaxF1 - 1 - r_.nsMinF1) * s_.mnsize);
}

if (r_.nsMinF == 0) {

HandOverMagneticAxis(m_h_, r1_e, z1_e, s_);
}

#ifdef _OPENMP
#pragma omp barrier
#endif
const double netToroidalCurrent = m_h_.cTor / MU_0;
const bool at_checkpoint_iteration =
iter2 >= iterations_before_checkpointing;

#ifdef _OPENMP
#pragma omp single
#endif
{
m_h_.vacuum_status = absl::OkStatus();
#ifdef _OPENMP
#pragma omp parallel num_threads(m_vac_num_threads_)
#endif
{
int vac_thread_id = 0;
#ifdef _OPENMP
vac_thread_id = omp_get_thread_num();

CHECK_EQ(omp_get_num_threads(), m_vac_num_threads_)
<< "Nested vacuum parallel region was not granted the requested "
"number of threads";
#endif
const absl::StatusOr<bool> rc = (*m_fb_vac_)[vac_thread_id]->update(
m_h_.rCC_LCFS, m_h_.rSS_LCFS, m_h_.rSC_LCFS, m_h_.rCS_LCFS,
m_h_.zSC_LCFS, m_h_.zCS_LCFS, m_h_.zCC_LCFS, m_h_.zSS_LCFS,
signOfJacobian, m_h_.rAxis, m_h_.zAxis, &(m_h_.bSubUVac),
&(m_h_.bSubVVac), netToroidalCurrent, ivacskip, checkpoint,
at_checkpoint_iteration);

if (!rc.ok()) {
#ifdef _OPENMP
#pragma omp critical
#endif
{
if (m_h_.vacuum_status.ok()) {
m_h_.vacuum_status = rc.status();
}
}
}

if (vac_thread_id == 0) {
m_h_.vacuum_reached_checkpoint = rc.ok() && *rc;
}
}
}

if (!m_h_.vacuum_status.ok() && verbose) {
#ifdef _OPENMP
#pragma omp single
#endif
std::cout << "WARNING: " << m_h_.vacuum_status.message() << "\n";
}
if (m_h_.vacuum_reached_checkpoint) {
return true;
}

#ifdef _OPENMP
#pragma omp single
#endif
{

if (m_vacuum_pressure_state_ == VacuumPressureState::kInitializing) {
m_vacuum_pressure_state_ = VacuumPressureState::kInitialized;

if (verbose) {

const double fac = 1.0e-6 / MU_0;
std::cout << "\n";
std::cout << absl::StrFormat(
"2*pi * a * -BPOL(vac) = %10.2e MA       R * BTOR(vacuum) = "
"%10.2e\n",
m_h_.bSubUVac * fac, m_h_.bSubVVac);
std::cout << absl::StrFormat(
"     TOROIDAL CURRENT = %10.2e MA       R * BTOR(plasma) = "
"%10.2e\n",
m_h_.cTor * fac, m_h_.rBtor);
}
}
}

if (m_h_.rBtor * m_h_.bSubVVac < 0.0) {

return absl::FailedPreconditionError(
"IdealMHDModel::update: rbtor and bsubvvac must have the same "
"sign - maybe flip the sign of phiedge or the sign of the coil "
"currents");
} else if (fabs((m_h_.cTor - m_h_.bSubUVac) / m_h_.rBtor) > 0.01) {
return absl::FailedPreconditionError(
"IdealMHDModel::update: VAC-VMEC I_TOR MISMATCH : BOUNDARY MAY "
"ENCLOSE EXT. COIL");
}

if (m_vacuum_pressure_state_ == VacuumPressureState::kInitialized) {
#ifdef _OPENMP
#pragma omp single
#endif
m_fc_.restart_reason = RestartReason::BAD_JACOBIAN;
m_need_restart = true;
} else {
m_need_restart = false;
}

if (r_.nsMaxF1 == m_fc_.ns) {

double edgePressure =
m_p_.evalMassProfile((m_fc_.ns - 1.5) / (m_fc_.ns - 1.0));
if (edgePressure != 0.0) {
edgePressure = m_p_.evalMassProfile(1.0) / edgePressure *
m_p_.presH[r_.nsMaxH - 1 - r_.nsMinH];
}

for (int kl = 0; kl < s_.nZnT; ++kl) {

insideTotalPressure[kl] =
1.5 * totalPressure[(r_.nsMaxH - 1 - r_.nsMinH) * s_.nZnT + kl] -
0.5 * totalPressure[(r_.nsMaxH - 2 - r_.nsMinH) * s_.nZnT + kl];

const int k = kl / s_.nThetaEff;
const int l = kl % s_.nThetaEff;
const int idx_lk = l * s_.nZeta + k;
double outsideEdgePressure =
m_h_.vacuum_magnetic_pressure[idx_lk] + edgePressure;

int idx_kl = (r_.nsMaxF1 - 1 - r_.nsMinF1) * s_.nZnT + kl;
rBSq[kl] = outsideEdgePressure * (r1_e[idx_kl] + r1_o[idx_kl]) /
m_fc_.deltaS;

delBSq[kl] = fabs(outsideEdgePressure - insideTotalPressure[kl]);

m_h_.edge_total_pressure[kl] = insideTotalPressure[kl];
}

if (m_vacuum_pressure_state_ == VacuumPressureState::kInitialized) {

for (int kl = 0; kl < s_.nZnT; ++kl) {
m_h_.initial_plasma_pressure_at_boundary[kl] =
totalPressure[(r_.nsMaxH - 1 - r_.nsMinH) * s_.nZnT + kl];
m_h_.initial_vacuum_pressure_at_boundary[kl] =
m_h_.vacuum_magnetic_pressure[kl];
}
}
}

if (checkpoint == VmecCheckpoint::RBSQ &&
iter2 >= iterations_before_checkpointing) {
return true;
}
}
}

effectiveConstraintForce();

deAliasConstraintForce();
if (checkpoint == VmecCheckpoint::ALIAS &&
iter2 >= iterations_before_checkpointing) {
return true;
}

computeMHDForces();

assembleTotalForces();
if (checkpoint == VmecCheckpoint::REALSPACE_FORCES &&
iter2 >= iterations_before_checkpointing) {
return true;
}

forcesToFourier(m_physical_f);
if (checkpoint == VmecCheckpoint::FWD_DFT_FORCES &&
iter2 >= iterations_before_checkpointing) {
return true;
}

m_physical_f.decomposeInto(m_decomposed_f, m_p_.scalxc);

m_decomposed_f.m1Constraint(1.0 / std::numbers::sqrt2, signOfJacobian);

const bool fix_m1_gauge =
always_fix_m1_gauge || m_fc.fsqz < 1.0e-6 || iter2 < 2;
if (fix_m1_gauge) {

m_decomposed_f.zeroZForceForM1();
}

if (s_.mpolGeometry < s_.mpol || s_.ntorGeometry < s_.ntor) {
m_decomposed_f.maskGeometryAbove(s_.mpolGeometry, s_.ntorGeometry);
}

if (checkpoint == VmecCheckpoint::PHYSICAL_FORCES &&
iter2 >= iterations_before_checkpointing) {
return true;
}

bool almost_converged = (m_fc.fsqr + m_fc.fsqz) < 1.0e-6;

bool hot_restart = (iter2 == 1 && m_vacuum_pressure_state_ ==
VacuumPressureState::kInitialized);
bool includeEdgeRZForces =
((iter2 - iter1) < 50 && (almost_converged || hot_restart));
Eigen::Vector3d localFResInvar;
localFResInvar.setZero();
m_decomposed_f.residuals(localFResInvar, includeEdgeRZForces);

evalFResInvar(localFResInvar);

if (checkpoint == VmecCheckpoint::INVARIANT_RESIDUALS &&
iter2 >= iterations_before_checkpointing) {
return true;
}

applyM1Preconditioner(m_decomposed_f);
if (checkpoint == VmecCheckpoint::APPLY_M1_PRECONDITIONER &&
iter2 >= iterations_before_checkpointing) {
return true;
}

assembleRZPreconditioner();
if (checkpoint == VmecCheckpoint::ASSEMBLE_RZ_PRECONDITIONER &&
iter2 >= iterations_before_checkpointing) {
return true;
}

absl::Status status = applyRZPreconditioner(m_decomposed_f);
if (!status.ok()) {
return status;
}

applyLambdaPreconditioner(m_decomposed_f);
if (checkpoint == VmecCheckpoint::APPLY_RADIAL_PRECONDITIONER &&
iter2 >= iterations_before_checkpointing) {
return true;
}

if (s_.mpolGeometry < s_.mpol || s_.ntorGeometry < s_.ntor) {
m_decomposed_f.maskGeometryAbove(s_.mpolGeometry, s_.ntorGeometry);
}

Eigen::Vector3d localFResPrecd;
localFResPrecd.setZero();
m_decomposed_f.residuals(localFResPrecd, true);

evalFResPrecd(localFResPrecd);

if (checkpoint == VmecCheckpoint::PRECONDITIONED_RESIDUALS &&
iter2 >= iterations_before_checkpointing) {
return true;
}

#ifdef _OPENMP
#pragma omp single
#endif
{
if (iter2 == 1 && (m_fc.fsqr + m_fc.fsqz + m_fc.fsql) > 1.0e2) {

m_fc.restart_reason = RestartReason::HUGE_INITIAL_FORCES;
}
}

return false;
}

void IdealMhdModel::geometryFromFourier(const FourierGeometry& physical_x) {

if (s_.lthreed) {
dft_FourierToReal_3d_symm(physical_x);
} else {
dft_FourierToReal_2d_symm(physical_x);
}

if (s_.lasym) {
if (s_.lthreed) {
dft_FourierToReal_3d_asymm(physical_x);
} else {
dft_FourierToReal_2d_asymm(physical_x);
}
symrzl();
}

for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {
for (int kl = 0; kl < s_.nZnT; ++kl) {
int idx_kl1 = (jF - r_.nsMinF1) * s_.nZnT + kl;
int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;
ruFull[idx_kl] =
ru_e[idx_kl1] + m_p_.sqrtSF[jF - r_.nsMinF1] * ru_o[idx_kl1];
zuFull[idx_kl] =
zu_e[idx_kl1] + m_p_.sqrtSF[jF - r_.nsMinF1] * zu_o[idx_kl1];
}
}

if (r_.nsMaxF1 == m_fc_.ns) {

const int outer_index = (m_fc_.ns - 1 - r_.nsMinF1) * s_.nZnT + 0;

const int inner_index =
(m_fc_.ns - 1 - r_.nsMinF1) * s_.nZnT + (s_.nThetaReduced - 1);
RadialExtent radial_extent = {
.r_outer = r1_e[outer_index] + r1_o[outer_index],
.r_inner = r1_e[inner_index] + r1_o[inner_index]};
m_h_.SetRadialExtent(radial_extent);
}
if (r_.nsMinF1 == 0) {

GeometricOffset geometric_offset = {.r_00 = r1_e[0], .z_00 = z1_e[0]};
m_h_.SetGeometricOffset(geometric_offset);
}
}

void IdealMhdModel::dft_FourierToReal_3d_symm(
const FourierGeometry& physical_x) {
auto geometry = RealSpaceGeometry{.r1_e = r1_e,
.r1_o = r1_o,
.ru_e = ru_e,
.ru_o = ru_o,
.rv_e = rv_e,
.rv_o = rv_o,
.z1_e = z1_e,
.z1_o = z1_o,
.zu_e = zu_e,
.zu_o = zu_o,
.zv_e = zv_e,
.zv_o = zv_o,
.lu_e = lu_e,
.lu_o = lu_o,
.lv_e = lv_e,
.lv_o = lv_o,
.rCon = rCon,
.zCon = zCon};

FourierToReal3DSymmFastPoloidal(physical_x, xmpq, r_, s_, m_p_, t_, geometry);
}

void IdealMhdModel::dft_FourierToReal_3d_asymm(
const FourierGeometry& physical_x) {
auto geometry = RealSpaceGeometry{.r1_e = r1_asym_e,
.r1_o = r1_asym_o,
.ru_e = ru_asym_e,
.ru_o = ru_asym_o,
.rv_e = rv_asym_e,
.rv_o = rv_asym_o,
.z1_e = z1_asym_e,
.z1_o = z1_asym_o,
.zu_e = zu_asym_e,
.zu_o = zu_asym_o,
.zv_e = zv_asym_e,
.zv_o = zv_asym_o,
.lu_e = lu_asym_e,
.lu_o = lu_asym_o,
.lv_e = lv_asym_e,
.lv_o = lv_asym_o,
.rCon = rCon_asym,
.zCon = zCon_asym};
FourierToReal3DAsymFastPoloidal(physical_x, xmpq, r_, s_, m_p_, t_, geometry);
}

void IdealMhdModel::dft_FourierToReal_2d_symm(
const FourierGeometry& physical_x) {

const int num_realsp = (r_.nsMaxF1 - r_.nsMinF1) * s_.nZnT;

for (auto* v :
{&r1_e, &r1_o, &ru_e, &ru_o, &z1_e, &z1_o, &zu_e, &zu_o, &lu_e, &lu_o}) {
absl::c_fill_n(*v, num_realsp, 0);
}

int num_con = (r_.nsMaxFIncludingLcfs - r_.nsMinF) * s_.nZnT;
absl::c_fill_n(rCon, num_con, 0);
absl::c_fill_n(zCon, num_con, 0);

#ifdef _OPENMP
#pragma omp barrier
#endif

for (int jF = r_.nsMinF1; jF < r_.nsMaxF1; ++jF) {
double* src_rcc = &(physical_x.rmncc[(jF - r_.nsMinF1) * s_.mnsize]);
double* src_zsc = &(physical_x.zmnsc[(jF - r_.nsMinF1) * s_.mnsize]);
double* src_lsc = &(physical_x.lmnsc[(jF - r_.nsMinF1) * s_.mnsize]);

for (int l = 0; l < s_.nThetaReduced; ++l) {
std::array<double, 2> rnkcc = {0.0, 0.0};
std::array<double, 2> rnkcc_m = {0.0, 0.0};
std::array<double, 2> znksc = {0.0, 0.0};
std::array<double, 2> znksc_m = {0.0, 0.0};
std::array<double, 2> lnksc_m = {0.0, 0.0};

int num_m = s_.mpol;
if (jF == 0) {

num_m = 2;
}

for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const int idx_ml = m * s_.nThetaReduced + l;
const double cosmu = t_.cosmu[idx_ml];
rnkcc[m_parity] += src_rcc[m] * cosmu;
}

for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const int idx_ml = m * s_.nThetaReduced + l;
const double sinmum = t_.sinmum[idx_ml];
rnkcc_m[m_parity] += src_rcc[m] * sinmum;
}

for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const int idx_ml = m * s_.nThetaReduced + l;
const double sinmu = t_.sinmu[idx_ml];
znksc[m_parity] += src_zsc[m] * sinmu;
}

for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const int idx_ml = m * s_.nThetaReduced + l;
const double cosmum = t_.cosmum[idx_ml];
znksc_m[m_parity] += src_zsc[m] * cosmum;
}

for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const int idx_ml = m * s_.nThetaReduced + l;
const double cosmum = t_.cosmum[idx_ml];
lnksc_m[m_parity] += src_lsc[m] * cosmum;
}

for (int k = 0; k < s_.nZeta; ++k) {
const int idx_jkl =
((jF - r_.nsMinF1) * s_.nZeta + k) * s_.nThetaEff + l;
r1_e[idx_jkl] += rnkcc[kEvenParity];
ru_e[idx_jkl] += rnkcc_m[kEvenParity];
z1_e[idx_jkl] += znksc[kEvenParity];
zu_e[idx_jkl] += znksc_m[kEvenParity];
lu_e[idx_jkl] += lnksc_m[kEvenParity];
r1_o[idx_jkl] += rnkcc[kOddParity];
ru_o[idx_jkl] += rnkcc_m[kOddParity];
z1_o[idx_jkl] += znksc[kOddParity];
zu_o[idx_jkl] += znksc_m[kOddParity];
lu_o[idx_jkl] += lnksc_m[kOddParity];
}
}
}

for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {
double* src_rcc = &(physical_x.rmncc[(jF - r_.nsMinF1) * s_.mnsize]);
double* src_zsc = &(physical_x.zmnsc[(jF - r_.nsMinF1) * s_.mnsize]);

int num_m = s_.mpol;
if (jF == 0) {

num_m = 2;
}

for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const double scale =
xmpq[m] * (1 - m_parity + m_parity * m_p_.sqrtSF[jF - r_.nsMinF1]);

for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_ml = m * s_.nThetaReduced + l;
const double cosmu = t_.cosmu[idx_ml];
const int idx_con =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
rCon[idx_con] += src_rcc[m] * cosmu * scale;
}
}

for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_ml = m * s_.nThetaReduced + l;
const double sinmu = t_.sinmu[idx_ml];
const int idx_con =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
zCon[idx_con] += src_zsc[m] * sinmu * scale;
}
}
}
}
}

void IdealMhdModel::dft_FourierToReal_2d_asymm(
const FourierGeometry& physical_x) {
const int num_realsp = (r_.nsMaxF1 - r_.nsMinF1) * s_.nZnT;

for (auto* v : {&r1_asym_e, &r1_asym_o, &ru_asym_e, &ru_asym_o, &z1_asym_e,
&z1_asym_o, &zu_asym_e, &zu_asym_o, &lu_asym_e, &lu_asym_o}) {
absl::c_fill_n(*v, num_realsp, 0);
}

int num_con = (r_.nsMaxFIncludingLcfs - r_.nsMinF) * s_.nZnT;
absl::c_fill_n(rCon_asym, num_con, 0);
absl::c_fill_n(zCon_asym, num_con, 0);

#ifdef _OPENMP
#pragma omp barrier
#endif

for (int jF = r_.nsMinF1; jF < r_.nsMaxF1; ++jF) {
double* src_rsc = &(physical_x.rmnsc[(jF - r_.nsMinF1) * s_.mnsize]);
double* src_zcc = &(physical_x.zmncc[(jF - r_.nsMinF1) * s_.mnsize]);
double* src_lcc = &(physical_x.lmncc[(jF - r_.nsMinF1) * s_.mnsize]);

for (int l = 0; l < s_.nThetaReduced; ++l) {
std::array<double, 2> rnksc = {0.0, 0.0};
std::array<double, 2> rnksc_m = {0.0, 0.0};
std::array<double, 2> znkcc = {0.0, 0.0};
std::array<double, 2> znkcc_m = {0.0, 0.0};
std::array<double, 2> lnkcc_m = {0.0, 0.0};

int num_m = s_.mpol;
if (jF == 0) {
num_m = 2;
}

for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const int idx_ml = m * s_.nThetaReduced + l;
rnksc[m_parity] += src_rsc[m] * t_.sinmu[idx_ml];
}
for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const int idx_ml = m * s_.nThetaReduced + l;
rnksc_m[m_parity] += src_rsc[m] * t_.cosmum[idx_ml];
}
for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const int idx_ml = m * s_.nThetaReduced + l;
znkcc[m_parity] += src_zcc[m] * t_.cosmu[idx_ml];
}
for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const int idx_ml = m * s_.nThetaReduced + l;
znkcc_m[m_parity] += src_zcc[m] * t_.sinmum[idx_ml];
}
for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const int idx_ml = m * s_.nThetaReduced + l;
lnkcc_m[m_parity] += src_lcc[m] * t_.sinmum[idx_ml];
}

for (int k = 0; k < s_.nZeta; ++k) {
const int idx_jkl =
((jF - r_.nsMinF1) * s_.nZeta + k) * s_.nThetaEff + l;
r1_asym_e[idx_jkl] += rnksc[kEvenParity];
ru_asym_e[idx_jkl] += rnksc_m[kEvenParity];
z1_asym_e[idx_jkl] += znkcc[kEvenParity];
zu_asym_e[idx_jkl] += znkcc_m[kEvenParity];
lu_asym_e[idx_jkl] += lnkcc_m[kEvenParity];
r1_asym_o[idx_jkl] += rnksc[kOddParity];
ru_asym_o[idx_jkl] += rnksc_m[kOddParity];
z1_asym_o[idx_jkl] += znkcc[kOddParity];
zu_asym_o[idx_jkl] += znkcc_m[kOddParity];
lu_asym_o[idx_jkl] += lnkcc_m[kOddParity];
}
}
}

for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {
double* src_rsc = &(physical_x.rmnsc[(jF - r_.nsMinF1) * s_.mnsize]);
double* src_zcc = &(physical_x.zmncc[(jF - r_.nsMinF1) * s_.mnsize]);

int num_m = s_.mpol;
if (jF == 0) {
num_m = 2;
}

for (int m = 0; m < num_m; ++m) {
const int m_parity = m % 2;
const double scale =
xmpq[m] * (1 - m_parity + m_parity * m_p_.sqrtSF[jF - r_.nsMinF1]);

for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_ml = m * s_.nThetaReduced + l;
const int idx_con =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
rCon_asym[idx_con] += src_rsc[m] * t_.sinmu[idx_ml] * scale;
}
}
for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_ml = m * s_.nThetaReduced + l;
const int idx_con =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
zCon_asym[idx_con] += src_zcc[m] * t_.cosmu[idx_ml] * scale;
}
}
}
}
}

void IdealMhdModel::symrzl() {

const int nThetaEff = s_.nThetaEff;
const bool lthreed = s_.lthreed;

for (int jF = r_.nsMinF1; jF < r_.nsMaxF1; ++jF) {
const int base = (jF - r_.nsMinF1) * s_.nZnT;

for (int k = 0; k < s_.nZeta; ++k) {
const int kRev = (s_.nZeta - k) % s_.nZeta;
for (int l = s_.nThetaReduced; l < s_.nThetaEven; ++l) {
const int jl = base + k * nThetaEff + l;
const int jlRev = base + kRev * nThetaEff + (s_.nThetaEven - l);

r1_e[jl] = r1_e[jlRev] - r1_asym_e[jlRev];
r1_o[jl] = r1_o[jlRev] - r1_asym_o[jlRev];
zu_e[jl] = zu_e[jlRev] - zu_asym_e[jlRev];
zu_o[jl] = zu_o[jlRev] - zu_asym_o[jlRev];
lu_e[jl] = lu_e[jlRev] - lu_asym_e[jlRev];
lu_o[jl] = lu_o[jlRev] - lu_asym_o[jlRev];

ru_e[jl] = -ru_e[jlRev] + ru_asym_e[jlRev];
ru_o[jl] = -ru_o[jlRev] + ru_asym_o[jlRev];
z1_e[jl] = -z1_e[jlRev] + z1_asym_e[jlRev];
z1_o[jl] = -z1_o[jlRev] + z1_asym_o[jlRev];

if (lthreed) {
zv_e[jl] = zv_e[jlRev] - zv_asym_e[jlRev];
zv_o[jl] = zv_o[jlRev] - zv_asym_o[jlRev];
lv_e[jl] = lv_e[jlRev] - lv_asym_e[jlRev];
lv_o[jl] = lv_o[jlRev] - lv_asym_o[jlRev];
rv_e[jl] = -rv_e[jlRev] + rv_asym_e[jlRev];
rv_o[jl] = -rv_o[jlRev] + rv_asym_o[jlRev];
}
}
}

for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int jl = base + k * nThetaEff + l;
r1_e[jl] += r1_asym_e[jl];
r1_o[jl] += r1_asym_o[jl];
ru_e[jl] += ru_asym_e[jl];
ru_o[jl] += ru_asym_o[jl];
z1_e[jl] += z1_asym_e[jl];
z1_o[jl] += z1_asym_o[jl];
zu_e[jl] += zu_asym_e[jl];
zu_o[jl] += zu_asym_o[jl];
lu_e[jl] += lu_asym_e[jl];
lu_o[jl] += lu_asym_o[jl];
if (lthreed) {
rv_e[jl] += rv_asym_e[jl];
rv_o[jl] += rv_asym_o[jl];
zv_e[jl] += zv_asym_e[jl];
zv_o[jl] += zv_asym_o[jl];
lv_e[jl] += lv_asym_e[jl];
lv_o[jl] += lv_asym_o[jl];
}
}
}
}

for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {
const int base = (jF - r_.nsMinF) * s_.nZnT;
for (int k = 0; k < s_.nZeta; ++k) {
const int kRev = (s_.nZeta - k) % s_.nZeta;
for (int l = s_.nThetaReduced; l < s_.nThetaEven; ++l) {
const int jl = base + k * nThetaEff + l;
const int jlRev = base + kRev * nThetaEff + (s_.nThetaEven - l);
rCon[jl] = rCon[jlRev] - rCon_asym[jlRev];
zCon[jl] = -zCon[jlRev] + zCon_asym[jlRev];
}
}
for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int jl = base + k * nThetaEff + l;
rCon[jl] += rCon_asym[jl];
zCon[jl] += zCon_asym[jl];
}
}
}
}

void IdealMhdModel::rzConIntoVolume() {

if (r_.nsMaxF1 == m_fc_.ns) {
for (int kl = 0; kl < s_.nZnT; ++kl) {
int lcfs_kl = (m_fc_.ns - 1 - r_.nsMinF) * s_.nZnT + kl;
m_h_.rCon_LCFS[kl] = rCon[lcfs_kl];
m_h_.zCon_LCFS[kl] = zCon[lcfs_kl];
}
}

#ifdef _OPENMP
#pragma omp barrier
#endif

for (int jF = std::max(1, r_.nsMinF); jF < r_.nsMaxFIncludingLcfs; ++jF) {
double sFull = m_p_.sqrtSF[jF - r_.nsMinF1] * m_p_.sqrtSF[jF - r_.nsMinF1];
for (int kl = 0; kl < s_.nZnT; ++kl) {
int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;
rCon0[idx_kl] = m_h_.rCon_LCFS[kl] * sFull;
zCon0[idx_kl] = m_h_.zCon_LCFS[kl] * sFull;
}
}
}

void IdealMhdModel::computeJacobian() {

ComputeHalfGridJacobian(
r1_e.data(), r1_o.data(), z1_e.data(), z1_o.data(), ru_e.data(),
ru_o.data(), zu_e.data(), zu_o.data(), m_p_.sqrtSH.data(), m_fc_.deltaS,
dSHalfDsInterp, s_.nZnT, r_.nsMinF1, r_.nsMinH, r_.nsMaxH, r12.data(),
ru12.data(), zu12.data(), rs.data(), zs.data(), tau.data());

double minTau = 0.0;
double maxTau = 0.0;
const int nTau = (r_.nsMaxH - r_.nsMinH) * s_.nZnT;
for (int i = 0; i < nTau; ++i) {
const double t = tau[i];
if (t < minTau || minTau == 0.0) {
minTau = t;
}
if (t > maxTau || maxTau == 0.0) {
maxTau = t;
}
}

bool localBadJacobian =
(minTau * maxTau < 0.0) || !std::isfinite(minTau * maxTau);

if (localBadJacobian) {
#ifdef _OPENMP
#pragma omp critical
#endif
{
m_fc_.restart_reason = RestartReason::BAD_JACOBIAN;
}
}
#ifdef _OPENMP
#pragma omp barrier
#endif
}

void IdealMhdModel::computeMetricElements() {

ComputeMetricElements(r1_e.data(), r1_o.data(), ru_e.data(), ru_o.data(),
zu_e.data(), zu_o.data(), rv_e.data(), rv_o.data(),
zv_e.data(), zv_o.data(), tau.data(), r12.data(),
m_p_.sqrtSF.data(), m_p_.sqrtSH.data(), s_.lthreed,
s_.nZnT, r_.nsMinF1, r_.nsMinH, r_.nsMaxH, gsqrt.data(),
guu.data(), guv.data(), gvv.data());
}

void IdealMhdModel::updateDifferentialVolume() {

for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
m_p_.dVdsH[jH - r_.nsMinH] = 0.0;
for (int kl = 0; kl < s_.nZnT; ++kl) {
int l = kl % s_.nThetaEff;

m_p_.dVdsH[jH - r_.nsMinH] +=
gsqrt[(jH - r_.nsMinH) * s_.nZnT + kl] * s_.wInt[l];
}

m_p_.dVdsH[jH - r_.nsMinH] *= signOfJacobian;
}
}

void IdealMhdModel::computeInitialVolume() {
double localPlasmaVolume = 0.0;
for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {

if (jH < r_.nsMaxH - 1 || jH == m_fc_.ns - 2) {
localPlasmaVolume += m_p_.dVdsH[jH - r_.nsMinH];
}
}
localPlasmaVolume *= m_fc_.deltaS;

const double localVolume = localPlasmaVolume * (2.0 * M_PI) * (2.0 * M_PI);
SumOverThreads(&localVolume, 1, r_.get_thread_id(), r_.get_num_threads(),
m_h_.thread_reduce_slots.data(), &m_h_.voli);
}

void IdealMhdModel::updateVolume() {
double localPlasmaVolume = 0.0;
for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {

if (jH < r_.nsMaxH - 1 || jH == m_fc_.ns - 2) {
localPlasmaVolume += m_p_.dVdsH[jH - r_.nsMinH];
}
}
localPlasmaVolume *= m_fc_.deltaS;

SumOverThreads(&localPlasmaVolume, 1, r_.get_thread_id(),
r_.get_num_threads(), m_h_.thread_reduce_slots.data(),
&m_h_.plasmaVolume);
}

void IdealMhdModel::computeBContra() {

{
const int j0 = r_.nsMinH;
for (int kl = 0; kl < s_.nZnT; ++kl) {
lu_e[(j0 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
lu_o[(j0 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
if (s_.lthreed) {
lv_e[(j0 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
lv_o[(j0 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
}
lu_e[(j0 - r_.nsMinF1) * s_.nZnT + kl] += m_p_.phipF[j0 - r_.nsMinF1];
}
for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
for (int kl = 0; kl < s_.nZnT; ++kl) {
lu_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
lu_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
if (s_.lthreed) {
lv_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
lv_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
}
lu_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl] +=
m_p_.phipF[jH + 1 - r_.nsMinH];
}
}
}

ComputeBsupContra(lu_e.data(), lu_o.data(), lv_e.data(), lv_o.data(),
gsqrt.data(), m_p_.sqrtSH.data(), s_.lthreed, s_.nZnT,
r_.nsMinF1, r_.nsMinH, r_.nsMaxH, bsupu.data(),
bsupv.data());

if (ncurr == 1) {

for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
double jvPlasma = 0.0;
double avg_guu_gsqrt = 0.0;
for (int kl = 0; kl < s_.nZnT; ++kl) {
int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;
int l = kl % s_.nThetaEff;
if (s_.lthreed) {
jvPlasma += (guu[iHalf] * bsupu[iHalf] + guv[iHalf] * bsupv[iHalf]) *
s_.wInt[l];
} else {
jvPlasma += guu[iHalf] * bsupu[iHalf] * s_.wInt[l];
}
avg_guu_gsqrt += guu[iHalf] / gsqrt[iHalf] * s_.wInt[l];
}

if (avg_guu_gsqrt != 0.0) {
m_p_.chipH[jH - r_.nsMinH] =
(m_p_.currH[jH - r_.nsMinH] - jvPlasma) / avg_guu_gsqrt;
}

if (m_p_.phipH[jH - r_.nsMinH] != 0.0) {
m_p_.iotaH[jH - r_.nsMinH] =
m_p_.chipH[jH - r_.nsMinH] / m_p_.phipH[jH - r_.nsMinH];
}
}
} else {

for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
m_p_.chipH[jH - r_.nsMinH] =
m_p_.iotaH[jH - r_.nsMinH] * m_p_.phipH[jH - r_.nsMinH];
}
}

if (r_.nsMinF1 == 0) {

m_p_.chipF[0] = 1.5 * m_p_.chipH[0] - 0.5 * m_p_.chipH[1];
}
for (int jFi = r_.nsMinFi; jFi < r_.nsMaxFi; ++jFi) {
m_p_.chipF[jFi - r_.nsMinF1] =
0.5 * (m_p_.chipH[jFi - r_.nsMinH] + m_p_.chipH[jFi - 1 - r_.nsMinH]);
}
if (r_.nsMaxF1 == m_fc_.ns) {

m_p_.chipF[r_.nsMaxF1 - 1 - r_.nsMinF1] =
1.5 * m_p_.chipH[r_.nsMaxH - 1 - r_.nsMinH] -
0.5 * m_p_.chipH[r_.nsMaxH - 2 - r_.nsMinH];
}

if (r_.nsMinF1 == 0) {
m_p_.iotaF[0] = 1.5 * m_p_.iotaH[0] - 0.5 * m_p_.iotaH[1];
}
for (int jFi = r_.nsMinFi; jFi < r_.nsMaxFi; ++jFi) {
m_p_.iotaF[jFi - r_.nsMinF1] =
0.5 * (m_p_.iotaH[jFi - r_.nsMinH] + m_p_.iotaH[jFi - 1 - r_.nsMinH]);
}
if (r_.nsMaxF1 == m_fc_.ns) {

m_p_.iotaF[r_.nsMaxF1 - 1 - r_.nsMinF1] =
1.5 * m_p_.iotaH[r_.nsMaxH - 1 - r_.nsMinH] -
0.5 * m_p_.iotaH[r_.nsMaxH - 2 - r_.nsMinH];
}

for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
for (int kl = 0; kl < s_.nZnT; ++kl) {
int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;
bsupu[iHalf] += m_p_.chipH[jH - r_.nsMinH] / gsqrt[iHalf];
}
}
}

void IdealMhdModel::computeBCo() {

ComputeBCo(guu.data(), guv.data(), gvv.data(), bsupu.data(), bsupv.data(),
s_.lthreed, static_cast<int>(bsupu.size()), bsubu.data(),
bsubv.data());
}

void IdealMhdModel::pressureAndEnergies() {

double localThermalEnergy = 0.0;
for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {

m_p_.presH[jH - r_.nsMinH] =
m_p_.massH[jH - r_.nsMinH] /
pow(m_p_.dVdsH[jH - r_.nsMinH], adiabaticIndex);

if (jH < r_.nsMaxH - 1 || jH == m_fc_.ns - 2) {
localThermalEnergy +=
m_p_.presH[jH - r_.nsMinH] * m_p_.dVdsH[jH - r_.nsMinH];
}
}

localThermalEnergy *= m_fc_.deltaS;

ComputeMagneticPressure(bsupu.data(), bsubu.data(), bsupv.data(),
bsubv.data(), static_cast<int>(bsupu.size()),
totalPressure.data());

double localMagneticEnergy = 0.0;
for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
const int offset = (jH - r_.nsMinH) * s_.nZnT;

if (jH < r_.nsMaxH - 1 || jH == m_fc_.ns - 2) {
for (int kl = 0; kl < s_.nZnT; ++kl) {
int l = kl % s_.nThetaEff;
localMagneticEnergy +=
gsqrt[offset + kl] * totalPressure[offset + kl] * s_.wInt[l];
}
}

totalPressure.segment(offset, s_.nZnT).array() +=
m_p_.presH[jH - r_.nsMinH];
}

localMagneticEnergy = fabs(localMagneticEnergy) * m_fc_.deltaS;

#ifdef _OPENMP
#pragma omp single
#endif
{
m_h_.thermalEnergy = 0.0;
m_h_.magneticEnergy = 0.0;
}

SumOverThreads(&localThermalEnergy, 1, r_.get_thread_id(),
r_.get_num_threads(), m_h_.thread_reduce_slots.data(),
&m_h_.thermalEnergy);
SumOverThreads(&localMagneticEnergy, 1, r_.get_thread_id(),
r_.get_num_threads(), m_h_.thread_reduce_slots.data(),
&m_h_.magneticEnergy);

#ifdef _OPENMP
#pragma omp single
#endif

m_h_.mhdEnergy =
m_h_.magneticEnergy + m_h_.thermalEnergy / (adiabaticIndex - 1.0);
#ifdef _OPENMP
#pragma omp barrier
#endif
}

void IdealMhdModel::radialForceBalance() {

for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
m_p_.bucoH[jH - r_.nsMinH] = 0.0;
m_p_.bvcoH[jH - r_.nsMinH] = 0.0;
for (int kl = 0; kl < s_.nZnT; ++kl) {
int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;
int l = kl % s_.nThetaEff;
m_p_.bucoH[jH - r_.nsMinH] += bsubu[iHalf] * s_.wInt[l];
m_p_.bvcoH[jH - r_.nsMinH] += bsubv[iHalf] * s_.wInt[l];
}
}

double signByDeltaS = signOfJacobian / m_fc_.deltaS;

for (int jFi = r_.nsMinFi; jFi < r_.nsMaxFi; ++jFi) {

m_p_.jcurvF[jFi - r_.nsMinFi] =
signByDeltaS *
(m_p_.bucoH[jFi - r_.nsMinH] - m_p_.bucoH[jFi - 1 - r_.nsMinH]);
m_p_.jcuruF[jFi - r_.nsMinFi] =
-signByDeltaS *
(m_p_.bvcoH[jFi - r_.nsMinH] - m_p_.bvcoH[jFi - 1 - r_.nsMinH]);

m_p_.presgradF[jFi - r_.nsMinFi] =
(m_p_.presH[jFi - r_.nsMinH] - m_p_.presH[jFi - 1 - r_.nsMinH]) /
m_fc_.deltaS;

m_p_.dVdsF[jFi - r_.nsMinFi] =
0.5 * (m_p_.dVdsH[jFi - r_.nsMinH] + m_p_.dVdsH[jFi - 1 - r_.nsMinH]);

m_p_.equiF[jFi - r_.nsMinFi] =
(m_p_.chipF[jFi - r_.nsMinF1] * m_p_.jcurvF[jFi - r_.nsMinFi] -
m_p_.phipF[jFi - r_.nsMinF1] * m_p_.jcuruF[jFi - r_.nsMinFi]) /
m_p_.dVdsF[jFi - r_.nsMinFi] +
m_p_.presgradF[jFi - r_.nsMinFi];
}
}

void IdealMhdModel::hybridLambdaForce() {
#ifdef _OPENMP
#pragma omp barrier
#endif

ComputeHybridLambdaForce(
bsubu.data(), bsubv.data(), gvv.data(), gsqrt.data(), guv.data(),
bsupu.data(), lu_e.data(), lu_o.data(), m_p_.sqrtSH.data(),
m_p_.sqrtSF.data(), m_p_.radialBlending.data(), constants_.lamscale,
s_.lthreed, s_.nZnT, r_.nsMinF, r_.nsMinF1, r_.nsMinH, r_.nsMaxH,
r_.nsMaxFIncludingLcfs, m_ls_.bsubu_i.data(), m_ls_.bsubv_i.data(),
m_ls_.gvv_gsqrt_i.data(), m_ls_.guv_bsupu_i.data(), blmn_e.data(),
blmn_o.data(), clmn_e.data(), clmn_o.data());

#ifdef _OPENMP
#pragma omp barrier
#endif
}

void IdealMhdModel::computeForceNorms(const FourierGeometry& decomposed_x) {

double energyDensity =
std::max(m_h_.magneticEnergy, m_h_.thermalEnergy) / m_h_.plasmaVolume;

double localForceNormSumRZ = 0.0;
double localForceNormSumL = 0.0;
for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
for (int kl = 0; kl < s_.nZnT; ++kl) {
int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;

if (jH < r_.nsMaxH - 1 || jH == m_fc_.ns - 2) {
int l = kl % s_.nThetaEff;
localForceNormSumRZ +=
guu[iHalf] * r12[iHalf] * r12[iHalf] * s_.wInt[l];
localForceNormSumL +=
(bsubu[iHalf] * bsubu[iHalf] + bsubv[iHalf] * bsubv[iHalf]) *
s_.wInt[l];
}
}
}

const int nsMinHere = r_.nsMinF;
double localForceNorm1 =
decomposed_x.rzNorm(false, nsMinHere, r_.nsMaxFIncludingLcfs);

#ifdef _OPENMP
#pragma omp single
#endif
{

m_h_.fNormRZ = 0.0;
m_h_.fNormL = 0.0;
m_h_.fNorm1 = 0.0;
}

SumOverThreads(&localForceNormSumRZ, 1, r_.get_thread_id(),
r_.get_num_threads(), m_h_.thread_reduce_slots.data(),
&m_h_.fNormRZ);
SumOverThreads(&localForceNormSumL, 1, r_.get_thread_id(),
r_.get_num_threads(), m_h_.thread_reduce_slots.data(),
&m_h_.fNormL);
SumOverThreads(&localForceNorm1, 1, r_.get_thread_id(), r_.get_num_threads(),
m_h_.thread_reduce_slots.data(), &m_h_.fNorm1);

#ifdef _OPENMP
#pragma omp single
#endif
{
m_h_.fNormRZ = 1.0 / (m_h_.fNormRZ * energyDensity * energyDensity);
m_h_.fNormL =
1.0 / (m_h_.fNormL * constants_.lamscale * constants_.lamscale);
m_h_.fNorm1 = 1.0 / m_h_.fNorm1;
}
#ifdef _OPENMP
#pragma omp barrier
#endif
}

void IdealMhdModel::computeMHDForces() {
int jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns - 1);
if (m_fc_.lfreeb) {
jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns);
}

ComputeMHDForceDensity(
r1_e.data(), r1_o.data(), ru_e.data(), ru_o.data(), zu_e.data(),
zu_o.data(), z1_o.data(), rv_e.data(), rv_o.data(), zv_e.data(),
zv_o.data(), r12.data(), ru12.data(), zu12.data(), rs.data(), zs.data(),
tau.data(), totalPressure.data(), gsqrt.data(), bsupu.data(),
bsupv.data(), m_p_.sqrtSF.data(), m_p_.sqrtSH.data(), m_ls_.P_i.data(),
m_ls_.rup_i.data(), m_ls_.zup_i.data(), m_ls_.rsp_i.data(),
m_ls_.zsp_i.data(), m_ls_.taup_i.data(), m_ls_.gbubu_i.data(),
m_ls_.gbubv_i.data(), m_ls_.gbvbv_i.data(), m_ls_.P_o.data(),
m_ls_.rup_o.data(), m_ls_.zup_o.data(), m_ls_.rsp_o.data(),
m_ls_.zsp_o.data(), m_ls_.taup_o.data(), m_ls_.gbubu_o.data(),
m_ls_.gbubv_o.data(), m_ls_.gbvbv_o.data(), m_ls_.P_avg.data(),
m_ls_.P_wavg.data(), m_ls_.gbubu_avg.data(), m_ls_.gbubu_wavg.data(),
m_ls_.gbvbv_avg.data(), m_ls_.gbvbv_wavg.data(), m_ls_.gbubv_avg.data(),
m_ls_.gbubv_wavg.data(), m_fc_.deltaS, s_.nZnT, r_.nsMinF, r_.nsMinF1,
r_.nsMinH, r_.nsMaxH, jMaxRZ, s_.lthreed, armn_e.data(), armn_o.data(),
azmn_e.data(), azmn_o.data(), brmn_e.data(), brmn_o.data(), bzmn_e.data(),
bzmn_o.data(), crmn_e.data(), crmn_o.data(), czmn_e.data(),
czmn_o.data());
}

bool IdealMhdModel::shouldUpdateRadialPreconditioner(int iter1,
int iter2) const {
return ((iter2 - iter1) % m_fc_.kPreconditionerUpdateInterval == 0);
}

void IdealMhdModel::updateRadialPreconditioner() {
updateLambdaPreconditioner();

computePreconditioningMatrix(zs, zu12, zu_e, zu_o, z1_o, arm, ard, brm, brd,
cxd, cos01, rzu_fac);

computePreconditioningMatrix(rs, ru12, ru_e, ru_o, r1_o, azm, azd, bzm, bzd,
cxd, sin01, rru_fac);

if (lforbal) {

for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
if (jF == 0 || jF >= m_fc_.ns - 1) {
continue;
}
const int i = jF - r_.nsMinF;
const double sj = m_p_.sqrtSF[jF - r_.nsMinF1];
rzu_fac[i] *= sj;
rru_fac[i] *= sj;
frcc_fac[i] = 1.0 / rzu_fac[i];
rzu_fac[i] *= 0.5;
fzsc_fac[i] = -1.0 / rru_fac[i];
rru_fac[i] *= 0.5;
}
}
}

void IdealMhdModel::updateLambdaPreconditioner() {

const double pFactor = kLambdaPreconditionerDampingFactor /
(4.0 * constants_.lamscale * constants_.lamscale);

for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
bLambda[jH + 1 - r_.nsMinH] = 0.0;
dLambda[jH + 1 - r_.nsMinH] = 0.0;
cLambda[jH + 1 - r_.nsMinH] = 0.0;
for (int kl = 0; kl < s_.nZnT; ++kl) {
int idx_kl = (jH - r_.nsMinH) * s_.nZnT + kl;
int l = kl % s_.nThetaEff;
bLambda[jH + 1 - r_.nsMinH] += guu[idx_kl] / gsqrt[idx_kl] * s_.wInt[l];
cLambda[jH + 1 - r_.nsMinH] += gvv[idx_kl] / gsqrt[idx_kl] * s_.wInt[l];
}

if (s_.lthreed) {
for (int kl = 0; kl < s_.nZnT; ++kl) {
int idx_kl = (jH - r_.nsMinH) * s_.nZnT + kl;
int l = kl % s_.nThetaEff;
dLambda[jH + 1 - r_.nsMinH] += guv[idx_kl] / gsqrt[idx_kl] * s_.wInt[l];
}
}
}

if (r_.nsMinF == 0) {
bLambda[0] = bLambda[1];
dLambda[0] = dLambda[1];
cLambda[0] = cLambda[1];
}

int jMin = 0;
if (r_.nsMinF == 0) {

jMin = 1;
}

for (int jF = std::max(jMin, r_.nsMinF); jF < r_.nsMaxFIncludingLcfs; ++jF) {
bLambda[jF - r_.nsMinF] =
0.5 * (bLambda[jF + 1 - r_.nsMinH] + bLambda[jF - r_.nsMinH]);
dLambda[jF - r_.nsMinF] =
0.5 * (dLambda[jF + 1 - r_.nsMinH] + dLambda[jF - r_.nsMinH]);
cLambda[jF - r_.nsMinF] =
0.5 * (cLambda[jF + 1 - r_.nsMinH] + cLambda[jF - r_.nsMinH]);
}

for (int jF = std::max(jMin, r_.nsMinF); jF < r_.nsMaxFIncludingLcfs; ++jF) {
for (int n = 0; n < s_.ntor + 1; ++n) {
double tnn = n * s_.nfp * n * s_.nfp;

for (int m = 0; m < s_.mpol; ++m) {
if (m == 0 && n == 0) {
continue;
}

int idx_mn = ((jF - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;

int tmm = m * m;

double pwr = std::min(tmm / (kLambdaHighMDampingReferenceM *
kLambdaHighMDampingReferenceM),
kLambdaHighMDampingMaxPower);
double tmn = 2.0 * m * n * s_.nfp;

double faclam =
tnn * bLambda[jF - r_.nsMinF] +
tmn * copysign(dLambda[jF - r_.nsMinF], bLambda[jF - r_.nsMinF]) +
tmm * cLambda[jF - r_.nsMinF];

if (faclam == 0.0) {
faclam = kLambdaPreconditionerZeroGuard;
}

lambdaPreconditioner[idx_mn] =
pFactor / faclam * pow(m_p_.sqrtSF[jF - r_.nsMinF1], pwr);
}
}
}
}

void IdealMhdModel::computePreconditioningMatrix(
const Eigen::VectorXd& xs, const Eigen::VectorXd& xu12,
const Eigen::VectorXd& xu_e, const Eigen::VectorXd& xu_o,
const Eigen::VectorXd& x1_o, Eigen::VectorXd& m_axm, Eigen::VectorXd& m_axd,
Eigen::VectorXd& m_bxm, Eigen::VectorXd& m_bxd, Eigen::VectorXd& m_cxd,
const Eigen::VectorXd& trigmult, Eigen::VectorXd& m_eqfactor) {

const bool do_eqfactor = m_eqfactor.size() > 0;
Eigen::VectorXd temp_h;
if (do_eqfactor) {
temp_h.setZero(r_.nsMaxH - r_.nsMinH);
}

double pFactor = -4.0;

absl::c_fill_n(ax, (r_.nsMaxH - r_.nsMinH) * 4, 0);
absl::c_fill_n(bx, (r_.nsMaxH - r_.nsMinH) * 3, 0);
absl::c_fill_n(cx, (r_.nsMaxH - r_.nsMinH), 0);

for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
for (int kl = 0; kl < s_.nZnT; ++kl) {
int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;
int iFull_0 = (jH - r_.nsMinF1) * s_.nZnT + kl;
int iFull_1 = (jH + 1 - r_.nsMinF1) * s_.nZnT + kl;

int l = kl % s_.nThetaEff;

double pTau =
pFactor * r12[iHalf] * totalPressure[iHalf] / tau[iHalf] * s_.wInt[l];

double t1a = xu12[iHalf] / m_fc_.deltaS;
double t2a =
0.25 * (xu_e[iFull_1] / m_p_.sqrtSH[jH - r_.nsMinH] + xu_o[iFull_1]) /
m_p_.sqrtSH[jH - r_.nsMinH];
double t3a =
0.25 * (xu_e[iFull_0] / m_p_.sqrtSH[jH - r_.nsMinH] + xu_o[iFull_0]) /
m_p_.sqrtSH[jH - r_.nsMinH];

ax[(jH - r_.nsMinH) * 4 + 0] += pTau * t1a * t1a;

ax[(jH - r_.nsMinH) * 4 + 1] += pTau * (t1a + t2a) * (-t1a + t3a);

ax[(jH - r_.nsMinH) * 4 + 2] += pTau * (t1a + t2a) * (t1a + t2a);

ax[(jH - r_.nsMinH) * 4 + 3] += pTau * (-t1a + t3a) * (-t1a + t3a);

double t1b =
0.5 * (xs[iHalf] + 0.5 / m_p_.sqrtSH[jH - r_.nsMinH] * x1_o[iFull_1]);
double t2b =
0.5 * (xs[iHalf] + 0.5 / m_p_.sqrtSH[jH - r_.nsMinH] * x1_o[iFull_0]);

bx[(jH - r_.nsMinH) * 3 + 0] += pTau * t1b * t2b;

bx[(jH - r_.nsMinH) * 3 + 1] += pTau * t1b * t1b;

bx[(jH - r_.nsMinH) * 3 + 2] += pTau * t2b * t2b;

cx[jH - r_.nsMinH] += 0.25 * pFactor * bsupv[iHalf] * bsupv[iHalf] *
gsqrt[iHalf] * s_.wInt[l];

if (do_eqfactor) {

temp_h[jH - r_.nsMinH] +=
pTau * tau[iHalf] * trigmult[kl] * xu12[iHalf];
}
}
}

const Eigen::VectorXd& sm = m_p_.sm;
const Eigen::VectorXd& sp = m_p_.sp;

for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {

m_axm[(jH - r_.nsMinH) * 2 + kEvenParity] = -ax[(jH - r_.nsMinH) * 4 + 0];
m_axm[(jH - r_.nsMinH) * 2 + kOddParity] =
ax[(jH - r_.nsMinH) * 4 + 1] * sm[jH - r_.nsMinH] * sp[jH - r_.nsMinH];

m_bxm[(jH - r_.nsMinH) * 2 + kEvenParity] = bx[(jH - r_.nsMinH) * 3 + 0];
m_bxm[(jH - r_.nsMinH) * 2 + kOddParity] =
bx[(jH - r_.nsMinH) * 3 + 0] * sm[jH - r_.nsMinH] * sp[jH - r_.nsMinH];
}

for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
int jH_i = jF - 1 - r_.nsMinH;
int jH_o = jF - r_.nsMinH;

m_axd[(jF - r_.nsMinF) * 2 + kEvenParity] =
(jF > 0 ? ax[jH_i * 4 + 0] : 0.0) +
(jF < m_fc_.ns - 1 ? ax[jH_o * 4 + 0] : 0.0);
m_axd[(jF - r_.nsMinF) * 2 + kOddParity] =
(jF > 0 ? ax[jH_i * 4 + 2] * sm[jH_i] * sm[jH_i] : 0.0) +
(jF < m_fc_.ns - 1 ? ax[jH_o * 4 + 3] * sp[jH_o] * sp[jH_o] : 0.0);

m_bxd[(jF - r_.nsMinF) * 2 + kEvenParity] =
(jF > 0 ? bx[jH_i * 3 + 1] : 0.0) +
(jF < m_fc_.ns - 1 ? bx[jH_o * 3 + 2] : 0.0);
m_bxd[(jF - r_.nsMinF) * 2 + kOddParity] =
(jF > 0 ? bx[jH_i * 3 + 1] * sm[jH_i] * sm[jH_i] : 0.0) +
(jF < m_fc_.ns - 1 ? bx[jH_o * 3 + 2] * sp[jH_o] * sp[jH_o] : 0.0);

m_cxd[jF - r_.nsMinF] =
(jF > 0 ? cx[jH_i] : 0.0) + (jF < m_fc_.ns - 1 ? cx[jH_o] : 0.0);
}

if (do_eqfactor) {

const double hs2 = m_fc_.deltaS * m_fc_.deltaS;
for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
const int jH_i = jF - 1 - r_.nsMinH;
const int jH_o = jF - r_.nsMinH;
double temp_f =
(jF > 0 ? temp_h[jH_i] / m_p_.dVdsH[jH_i] : 0.0) +
(jF < m_fc_.ns - 1 ? temp_h[jH_o] / m_p_.dVdsH[jH_o] : 0.0);
temp_f *= signOfJacobian;
const double axd_m1 = m_axd[(jF - r_.nsMinF) * 2 + kOddParity];
m_eqfactor[jF - r_.nsMinF] =
(jF > 0 && jF < m_fc_.ns - 1 && temp_f != 0.0) ? axd_m1 * hs2 / temp_f
: 0.0;
}
}
}

double IdealMhdModel::constraintMultiplierScale() const {

const double tcon_multiplier =
tcon0 * (1.0 + m_fc_.ns * (1.0 / 60.0 + m_fc_.ns / (200.0 * 120.0)));

return tcon_multiplier / (4.0 * 4.0);
}

absl::Status IdealMhdModel::constraintForceMultiplier() {

const double tcon_multiplier = constraintMultiplierScale();

int jMin = 0;
if (r_.nsMinF == 0) {
jMin = 1;
}

for (int jF = std::max(jMin, r_.nsMinF); jF < r_.nsMaxF; ++jF) {
double arNorm = 0.0;
double azNorm = 0.0;
for (int kl = 0; kl < s_.nZnT; ++kl) {
int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;
int l = kl % s_.nThetaEff;
arNorm += ruFull[idx_kl] * ruFull[idx_kl] * s_.wInt[l];
azNorm += zuFull[idx_kl] * zuFull[idx_kl] * s_.wInt[l];
}

if (arNorm == 0.0) {

return absl::FailedPreconditionError("arNorm should never be 0.0.");
}
if (azNorm == 0.0) {
return absl::FailedPreconditionError("azNorm should never be 0.0.");
}

double tcon_base =
std::min(fabs(ard[(jF - r_.nsMinF) * 2 + kEvenParity] / arNorm),
fabs(azd[(jF - r_.nsMinF) * 2 + kEvenParity] / azNorm));

tcon[jF - r_.nsMinF] =
tcon_base * tcon_multiplier * 32 * m_fc_.deltaS * 32 * m_fc_.deltaS;
}

if (r_.nsMaxF1 == m_fc_.ns) {

tcon[r_.nsMaxF1 - 1 - r_.nsMinF] = 0.5 * tcon[r_.nsMaxF1 - 2 - r_.nsMinF];
}

return absl::OkStatus();
}

void IdealMhdModel::effectiveConstraintForce() {

ComputeEffectiveConstraintForce(rCon.data(), rCon0.data(), zCon.data(),
zCon0.data(), ruFull.data(), zuFull.data(),
s_.nZnT, r_.nsMinF, r_.nsMaxFIncludingLcfs,
gConEff.data());
}

void IdealMhdModel::deAliasConstraintForce() {
vmecpp::deAliasConstraintForce(r_, t_, s_, faccon, tcon, gConEff, gsc, gcs,
gcc, gss, gConAsym, refl, gCon);
}

void IdealMhdModel::assembleTotalForces() {
#ifdef _OPENMP
#pragma omp barrier
#endif

if (m_fc_.lfreeb &&
m_vacuum_pressure_state_ >= VacuumPressureState::kInitialized &&
r_.nsMaxF1 == m_fc_.ns) {
for (int kl = 0; kl < s_.nZnT; ++kl) {
int idx_kl = (r_.nsMaxF - 1 - r_.nsMinF) * s_.nZnT + kl;

armn_e[idx_kl] += zuFull[idx_kl] * rBSq[kl];
armn_o[idx_kl] += zuFull[idx_kl] * rBSq[kl];
azmn_e[idx_kl] -= ruFull[idx_kl] * rBSq[kl];
azmn_o[idx_kl] -= ruFull[idx_kl] * rBSq[kl];
}
}

AddConstraintForces(rCon.data(), rCon0.data(), zCon.data(), zCon0.data(),
ruFull.data(), zuFull.data(), gCon.data(),
m_p_.sqrtSF.data(), s_.nZnT, r_.nsMinF, r_.nsMinF1,
r_.nsMaxF, brmn_e.data(), brmn_o.data(), bzmn_e.data(),
bzmn_o.data(), frcon_e.data(), frcon_o.data(),
fzcon_e.data(), fzcon_o.data());
}

#ifdef VMECPP_ENABLE_ENZYME
void IdealMhdModel::packGeometry(FourierGeometry& m_decomposed,
FourierGeometry& m_physical_scratch,
double* out, int gS, bool primal) {

m_decomposed.decomposeInto(m_physical_scratch, m_p_.scalxc);
m_physical_scratch.m1Constraint(1.0, signOfJacobian);
m_physical_scratch.extrapolateTowardsAxis();
geometryFromFourier(m_physical_scratch);

auto blk = [&](int b, const Eigen::VectorXd& src) {
const int n = static_cast<int>(src.size());
for (int i = 0; i < n; ++i) out[b * gS + i] = src[i];
};
blk(0, r1_e);
blk(1, r1_o);
blk(2, z1_e);
blk(3, z1_o);
blk(4, ru_e);
blk(5, ru_o);
blk(6, zu_e);
blk(7, zu_o);
blk(8, rv_e);
blk(9, rv_o);
blk(10, zv_e);
blk(11, zv_o);

auto blk_lam = [&](int b, const Eigen::VectorXd& src) {
const int n = static_cast<int>(src.size());
for (int i = 0; i < n; ++i) out[b * gS + i] = constants_.lamscale * src[i];
};
blk_lam(12, lu_e);
blk_lam(13, lu_o);
blk_lam(14, lv_e);
blk_lam(15, lv_o);
if (primal) {
const int nFullSurf = static_cast<int>(lu_e.size()) / s_.nZnT;
for (int jF = 0; jF < nFullSurf; ++jF) {
const double phip = m_p_.phipF[jF];
for (int kl = 0; kl < s_.nZnT; ++kl) {
out[12 * gS + jF * s_.nZnT + kl] += phip;
}
}
}
blk(16, rCon);
blk(17, zCon);
blk(18, ruFull);
blk(19, zuFull);
}

LocalForceComposition IdealMhdModel::makeLocalForceComposition(
int geom_stride) {
LocalForceComposition comp;
comp.nZnT = s_.nZnT;
comp.geom_stride = geom_stride;
comp.force_stride = (r_.nsMaxFIncludingLcfs - r_.nsMinF) * s_.nZnT;
comp.nsMinF = r_.nsMinF;
comp.nsMinF1 = r_.nsMinF1;
comp.nsMinH = r_.nsMinH;
comp.nsMaxH = r_.nsMaxH;
comp.jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns - 1);
comp.nsMaxFIncludingLcfs = r_.nsMaxFIncludingLcfs;
comp.sqrtSF = m_p_.sqrtSF.data();
comp.sqrtSH = m_p_.sqrtSH.data();
comp.chipH = m_p_.chipH.data();
comp.presH = m_p_.presH.data();
comp.radialBlending = m_p_.radialBlending.data();
comp.deltaS = m_fc_.deltaS;
comp.dSHalfDsInterp = dSHalfDsInterp;
comp.lamscale = constants_.lamscale;
comp.lthreed = s_.lthreed;
comp.with_constraint = true;
comp.lasym = s_.lasym;
comp.nsMaxF = r_.nsMaxF;
comp.nZeta = s_.nZeta;
comp.nThetaEff = s_.nThetaEff;
comp.ncurr = ncurr;
comp.currH = m_p_.currH.data();
comp.wInt = s_.wInt.data();
comp.nThetaEven = s_.nThetaEven;
comp.nThetaReduced = s_.nThetaReduced;
comp.mpol = s_.mpol;
comp.ntor = s_.ntor;
comp.nnyq2 = s_.nnyq2;
comp.rCon0 = rCon0.data();
comp.zCon0 = zCon0.data();
comp.faccon = faccon.data();
comp.ns = m_fc_.ns;
comp.tcon_multiplier = constraintMultiplierScale();
comp.sinmui = t_.sinmui.data();
comp.cosmui = t_.cosmui.data();
comp.cosnv = t_.cosnv.data();
comp.sinnv = t_.sinnv.data();
comp.sinmu = t_.sinmu.data();
comp.cosmu = t_.cosmu.data();
return comp;
}

void IdealMhdModel::applyExactForceJacobian(const double* geomP,
const double* dgeom,
int geom_stride,
FourierForces& m_physical_f,
FourierForces& m_decomposed_hv,
bool fix_m1_gauge) {
LocalForceComposition comp = makeLocalForceComposition(geom_stride);
const int nForce = comp.force_stride;

const int nWork = LocalForceWorkSize(comp);
std::vector<double> work(nWork, 0.0);
std::vector<double> dwork(nWork, 0.0);
std::vector<double> force(kLocalForceBlocks * nForce, 0.0);
std::vector<double> dforce(kLocalForceBlocks * nForce, 0.0);

ExactForceDensityJvp(geomP, dgeom, work.data(), dwork.data(), force.data(),
dforce.data(), &comp);

auto put = [&](int block, Eigen::VectorXd& dst) {
const int n = static_cast<int>(dst.size());
for (int i = 0; i < n; ++i) {
dst[i] = dforce[block * nForce + i];
}
};
put(0, armn_e);
put(1, armn_o);
put(2, azmn_e);
put(3, azmn_o);
put(4, brmn_e);
put(5, brmn_o);
put(6, bzmn_e);
put(7, bzmn_o);
put(12, blmn_e);
put(13, blmn_o);
if (s_.lthreed) {
put(8, crmn_e);
put(9, crmn_o);
put(10, czmn_e);
put(11, czmn_o);
put(14, clmn_e);
put(15, clmn_o);
}
put(16, frcon_e);
put(17, frcon_o);
put(18, fzcon_e);
put(19, fzcon_o);

forcesToFourier(m_physical_f);
m_physical_f.decomposeInto(m_decomposed_hv, m_p_.scalxc);
m_decomposed_hv.m1Constraint(1.0 / std::numbers::sqrt2, signOfJacobian);
if (fix_m1_gauge) {
m_decomposed_hv.zeroZForceForM1();
}
}

void IdealMhdModel::exactForceDensityTangent(const double* geomP,
const double* dgeom,
int geom_stride,
double* dforce_out) {
LocalForceComposition comp = makeLocalForceComposition(geom_stride);
const int nForce = comp.force_stride;
const int nWork = LocalForceWorkSize(comp);
std::vector<double> work(nWork, 0.0);
std::vector<double> dwork(nWork, 0.0);
std::vector<double> force(kLocalForceBlocks * nForce, 0.0);
ExactForceDensityJvp(geomP, dgeom, work.data(), dwork.data(), force.data(),
dforce_out, &comp);
}

void IdealMhdModel::exactForceDensityCotangent(const double* geomP,
const double* force_bar,
int geom_stride,
double* geom_bar_out) {
LocalForceComposition comp = makeLocalForceComposition(geom_stride);
const int nForce = comp.force_stride;
const int nWork = LocalForceWorkSize(comp);
std::vector<double> work(nWork, 0.0);
std::vector<double> work_bar(nWork, 0.0);
std::vector<double> force(kLocalForceBlocks * nForce, 0.0);

std::vector<double> fbar(force_bar, force_bar + kLocalForceBlocks * nForce);
ExactForceDensityVjp(geomP, geom_bar_out, work.data(), work_bar.data(),
force.data(), fbar.data(), &comp);
}

#endif

void IdealMhdModel::dft_ForcesToFourierTranspose_2d_symm(
const FourierForces& m_coeff_bar) {
for (auto* v :
{&armn_e, &armn_o, &brmn_e, &brmn_o, &azmn_e, &azmn_o, &bzmn_e, &bzmn_o,
&frcon_e, &frcon_o, &fzcon_e, &fzcon_o, &blmn_e, &blmn_o}) {
v->setZero();
}
int jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns - 1);
if (m_fc_.lfreeb &&
m_vacuum_pressure_state_ >= VacuumPressureState::kInitialized) {
jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns);
}
for (int jF = r_.nsMinF; jF < jMaxRZ; ++jF) {
const int num_m = (jF == 0) ? 1 : s_.mpol;
for (int m = 0; m < num_m; ++m) {
const bool m_even = m % 2 == 0;
auto& armn = m_even ? armn_e : armn_o;
auto& brmn = m_even ? brmn_e : brmn_o;
auto& azmn = m_even ? azmn_e : azmn_o;
auto& bzmn = m_even ? bzmn_e : bzmn_o;
auto& frcon = m_even ? frcon_e : frcon_o;
auto& fzcon = m_even ? fzcon_e : fzcon_o;
const int idx_jm = (jF - r_.nsMinF) * s_.mpol + m;
const double fr = m_coeff_bar.frcc[idx_jm];
const double fz = m_coeff_bar.fzsc[idx_jm];
for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_jl =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
const int idx_ml = m * s_.nThetaReduced + l;
const double cosmui = t_.cosmui[idx_ml];
const double sinmumi = t_.sinmumi[idx_ml];
const double sinmui = t_.sinmui[idx_ml];
const double cosmumi = t_.cosmumi[idx_ml];
armn[idx_jl] += fr * cosmui;
brmn[idx_jl] += fr * sinmumi;
frcon[idx_jl] += fr * xmpq[m] * cosmui;
azmn[idx_jl] += fz * sinmui;
bzmn[idx_jl] += fz * cosmumi;
fzcon[idx_jl] += fz * xmpq[m] * sinmui;
}
}
}
}
for (int jF = std::max(1, r_.nsMinF); jF < r_.nsMaxFIncludingLcfs; ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
const bool m_even = m % 2 == 0;
auto& blmn = m_even ? blmn_e : blmn_o;
const int idx_jm = (jF - r_.nsMinF) * s_.mpol + m;
const double fl = m_coeff_bar.flsc[idx_jm];
for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_jl =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
const double cosmumi = t_.cosmumi[m * s_.nThetaReduced + l];
blmn[idx_jl] += fl * cosmumi;
}
}
}
}
}

void IdealMhdModel::dft_FourierToRealTranspose_2d_symm(
FourierGeometry& m_coeff_bar_out) {
m_coeff_bar_out.setZero();
for (int jF = r_.nsMinF1; jF < r_.nsMaxF1; ++jF) {
double* dst_rcc = &(m_coeff_bar_out.rmncc[(jF - r_.nsMinF1) * s_.mnsize]);
double* dst_zsc = &(m_coeff_bar_out.zmnsc[(jF - r_.nsMinF1) * s_.mnsize]);
double* dst_lsc = &(m_coeff_bar_out.lmnsc[(jF - r_.nsMinF1) * s_.mnsize]);
for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_jl =
((jF - r_.nsMinF1) * s_.nZeta + k) * s_.nThetaEff + l;
const double r1eb = r1_e[idx_jl], rueb = ru_e[idx_jl],
z1eb = z1_e[idx_jl], zueb = zu_e[idx_jl],
lueb = lu_e[idx_jl];
const double r1ob = r1_o[idx_jl], ruob = ru_o[idx_jl],
z1ob = z1_o[idx_jl], zuob = zu_o[idx_jl],
luob = lu_o[idx_jl];
const int num_m = (jF == 0) ? 2 : s_.mpol;
for (int m = 0; m < num_m; ++m) {
const int p = m % 2;
const int idx_ml = m * s_.nThetaReduced + l;
const double cosmu = t_.cosmu[idx_ml];
const double sinmum = t_.sinmum[idx_ml];
const double sinmu = t_.sinmu[idx_ml];
const double cosmum = t_.cosmum[idx_ml];
dst_rcc[m] += (p ? r1ob : r1eb) * cosmu + (p ? ruob : rueb) * sinmum;
dst_zsc[m] += (p ? z1ob : z1eb) * sinmu + (p ? zuob : zueb) * cosmum;
dst_lsc[m] += (p ? luob : lueb) * cosmum;
}
}
}
}
for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {
double* dst_rcc = &(m_coeff_bar_out.rmncc[(jF - r_.nsMinF1) * s_.mnsize]);
double* dst_zsc = &(m_coeff_bar_out.zmnsc[(jF - r_.nsMinF1) * s_.mnsize]);
const int num_m = (jF == 0) ? 2 : s_.mpol;
for (int m = 0; m < num_m; ++m) {
const int p = m % 2;
const double scale = xmpq[m] * (1 - p + p * m_p_.sqrtSF[jF - r_.nsMinF1]);
for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_ml = m * s_.nThetaReduced + l;
const int idx_con =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
dst_rcc[m] += rCon[idx_con] * t_.cosmu[idx_ml] * scale;
dst_zsc[m] += zCon[idx_con] * t_.sinmu[idx_ml] * scale;
}
}
}
}
}

void IdealMhdModel::dft_ForcesToFourierTranspose_3d_symm(
const FourierForces& m_coeff_bar) {
for (auto* v :
{&armn_e, &armn_o, &azmn_e,  &azmn_o,  &blmn_e,  &blmn_o, &brmn_e,
&brmn_o, &bzmn_e, &bzmn_o,  &clmn_e,  &clmn_o,  &crmn_e, &crmn_o,
&czmn_e, &czmn_o, &frcon_e, &frcon_o, &fzcon_e, &fzcon_o}) {
v->setZero();
}
const int nThR = s_.nThetaReduced;
const int ntorp1 = s_.ntor + 1;
int jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns - 1);
if (m_fc_.lfreeb &&
m_vacuum_pressure_state_ >= VacuumPressureState::kInitialized) {
jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns);
}
const int jMinL = 1;
for (int jF = r_.nsMinF; jF < jMaxRZ; ++jF) {
const int mmax = (jF == 0) ? 1 : s_.mpol;
for (int m = 0; m < mmax; ++m) {
const bool m_even = m % 2 == 0;
auto& armn = m_even ? armn_e : armn_o;
auto& azmn = m_even ? azmn_e : azmn_o;
auto& blmn = m_even ? blmn_e : blmn_o;
auto& brmn = m_even ? brmn_e : brmn_o;
auto& bzmn = m_even ? bzmn_e : bzmn_o;
auto& clmn = m_even ? clmn_e : clmn_o;
auto& crmn = m_even ? crmn_e : crmn_o;
auto& czmn = m_even ? czmn_e : czmn_o;
auto& frcon = m_even ? frcon_e : frcon_o;
auto& fzcon = m_even ? fzcon_e : fzcon_o;
const int idx_ml_base = m * nThR;
for (int k = 0; k < s_.nZeta; ++k) {
const int idx_kl_base =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff;
const int idx_kn_base = k * (s_.nnyq2 + 1);
const int idx_mn_base = ((jF - r_.nsMinF) * s_.mpol + m) * ntorp1;
double rmkcc = 0, rmkss = 0, zmksc = 0, zmkcs = 0, rmkcc_n = 0,
zmkcs_n = 0, rmkss_n = 0, zmksc_n = 0, lmksc = 0, lmkcs = 0,
lmkcs_n = 0, lmksc_n = 0;
for (int nn = 0; nn < ntorp1; ++nn) {
const int kn = idx_kn_base + nn;
const int mn = idx_mn_base + nn;
const double cosnv = t_.cosnv[kn], sinnv = t_.sinnv[kn],
cosnvn = t_.cosnvn[kn], sinnvn = t_.sinnvn[kn];
const double frcc = m_coeff_bar.frcc[mn], frss = m_coeff_bar.frss[mn],
fzsc = m_coeff_bar.fzsc[mn], fzcs = m_coeff_bar.fzcs[mn];
rmkcc += frcc * cosnv;
rmkcc_n += frcc * sinnvn;
rmkss += frss * sinnv;
rmkss_n += frss * cosnvn;
zmksc += fzsc * cosnv;
zmksc_n += fzsc * sinnvn;
zmkcs += fzcs * sinnv;
zmkcs_n += fzcs * cosnvn;
if (jMinL <= jF) {
const double flsc = m_coeff_bar.flsc[mn],
flcs = m_coeff_bar.flcs[mn];
lmksc += flsc * cosnv;
lmksc_n += flsc * sinnvn;
lmkcs += flcs * sinnv;
lmkcs_n += flcs * cosnvn;
}
}
for (int l = 0; l < nThR; ++l) {
const int im = idx_ml_base + l;
const int kl = idx_kl_base + l;
const double cosmui = t_.cosmui[im], sinmui = t_.sinmui[im],
cosmumi = t_.cosmumi[im], sinmumi = t_.sinmumi[im];
const double tR = rmkcc * cosmui + rmkss * sinmui;
armn[kl] += tR;
frcon[kl] += xmpq[m] * tR;
brmn[kl] += rmkcc * sinmumi + rmkss * cosmumi;
const double tZ = zmksc * sinmui + zmkcs * cosmui;
azmn[kl] += tZ;
fzcon[kl] += xmpq[m] * tZ;
bzmn[kl] += zmksc * cosmumi + zmkcs * sinmumi;
crmn[kl] += -(rmkcc_n * cosmui + rmkss_n * sinmui);
czmn[kl] += -(zmkcs_n * cosmui + zmksc_n * sinmui);
blmn[kl] += lmksc * cosmumi + lmkcs * sinmumi;
clmn[kl] += -(lmkcs_n * cosmui + lmksc_n * sinmui);
}
}
}
}
for (int jF = jMaxRZ; jF < r_.nsMaxFIncludingLcfs; ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
const bool m_even = m % 2 == 0;
auto& blmn = m_even ? blmn_e : blmn_o;
auto& clmn = m_even ? clmn_e : clmn_o;
const int idx_ml_base = m * nThR;
for (int k = 0; k < s_.nZeta; ++k) {
const int idx_kl_base =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff;
const int idx_kn_base = k * (s_.nnyq2 + 1);
const int idx_mn_base = ((jF - r_.nsMinF) * s_.mpol + m) * ntorp1;
double lmksc = 0, lmkcs = 0, lmkcs_n = 0, lmksc_n = 0;
for (int nn = 0; nn < ntorp1; ++nn) {
const int kn = idx_kn_base + nn;
const int mn = idx_mn_base + nn;
lmksc += m_coeff_bar.flsc[mn] * t_.cosnv[kn];
lmksc_n += m_coeff_bar.flsc[mn] * t_.sinnvn[kn];
lmkcs += m_coeff_bar.flcs[mn] * t_.sinnv[kn];
lmkcs_n += m_coeff_bar.flcs[mn] * t_.cosnvn[kn];
}
for (int l = 0; l < nThR; ++l) {
const int im = idx_ml_base + l;
const int kl = idx_kl_base + l;
blmn[kl] += lmksc * t_.cosmumi[im] + lmkcs * t_.sinmumi[im];
clmn[kl] += -(lmkcs_n * t_.cosmui[im] + lmksc_n * t_.sinmui[im]);
}
}
}
}
}

void IdealMhdModel::dft_FourierToRealTranspose_3d_symm(
FourierGeometry& m_coeff_bar_out) {
m_coeff_bar_out.setZero();
const int nThR = s_.nThetaReduced;
const int ntorp1 = s_.ntor + 1;
for (int jF = r_.nsMinF1; jF < r_.nsMaxF1; ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
const bool m_even = m % 2 == 0;
const double con_factor =
m_even ? xmpq[m] : xmpq[m] * m_p_.sqrtSF[jF - r_.nsMinF1];
auto& r1 = m_even ? r1_e : r1_o;
auto& ru = m_even ? ru_e : ru_o;
auto& rv = m_even ? rv_e : rv_o;
auto& z1 = m_even ? z1_e : z1_o;
auto& zu = m_even ? zu_e : zu_o;
auto& zv = m_even ? zv_e : zv_o;
auto& lu = m_even ? lu_e : lu_o;
auto& lv = m_even ? lv_e : lv_o;
const int jMin = (m == 0 || m == 1) ? 0 : 1;
if (jF < jMin) {
continue;
}
const int idx_ml_base = m * nThR;
for (int k = 0; k < s_.nZeta; ++k) {
const int idx_kl_base =
((jF - r_.nsMinF1) * s_.nZeta + k) * s_.nThetaEff;
const bool con_in_range =
(r_.nsMinF <= jF && jF < r_.nsMaxFIncludingLcfs);
const int idx_con_base =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff;
double rmkcc = 0, rmkss = 0, rmkcc_n = 0, rmkss_n = 0, zmksc = 0,
zmkcs = 0, zmksc_n = 0, zmkcs_n = 0, lmksc = 0, lmkcs = 0,
lmksc_n = 0, lmkcs_n = 0;
for (int l = 0; l < nThR; ++l) {
const int im = idx_ml_base + l;
const int kl = idx_kl_base + l;
const double cosmu = t_.cosmu[im], sinmu = t_.sinmu[im],
sinmum = t_.sinmum[im], cosmum = t_.cosmum[im];
const double r1b = r1[kl], rub = ru[kl], rvb = rv[kl], z1b = z1[kl],
zub = zu[kl], zvb = zv[kl], lub = lu[kl], lvb = lv[kl];
double rConb = 0, zConb = 0;
if (con_in_range) {
rConb = rCon[idx_con_base + l];
zConb = zCon[idx_con_base + l];
}
rmkcc += r1b * cosmu + rub * sinmum + rConb * cosmu * con_factor;
rmkss += r1b * sinmu + rub * cosmum + rConb * sinmu * con_factor;
rmkcc_n += rvb * cosmu;
rmkss_n += rvb * sinmu;
zmksc += z1b * sinmu + zub * cosmum + zConb * sinmu * con_factor;
zmkcs += z1b * cosmu + zub * sinmum + zConb * cosmu * con_factor;
zmksc_n += zvb * sinmu;
zmkcs_n += zvb * cosmu;
lmksc += lub * cosmum;
lmkcs += lub * sinmum;
lmksc_n += -lvb * sinmu;
lmkcs_n += -lvb * cosmu;
}
const int idx_kn_base = k * (s_.nnyq2 + 1);
const int idx_mn_base = ((jF - r_.nsMinF1) * s_.mpol + m) * ntorp1;
for (int nn = 0; nn < ntorp1; ++nn) {
const int kn = idx_kn_base + nn;
const int mn = idx_mn_base + nn;
const double cosnv = t_.cosnv[kn], sinnv = t_.sinnv[kn],
cosnvn = t_.cosnvn[kn], sinnvn = t_.sinnvn[kn];
m_coeff_bar_out.rmncc[mn] += rmkcc * cosnv + rmkcc_n * sinnvn;
m_coeff_bar_out.rmnss[mn] += rmkss * sinnv + rmkss_n * cosnvn;
m_coeff_bar_out.zmnsc[mn] += zmksc * cosnv + zmksc_n * sinnvn;
m_coeff_bar_out.zmncs[mn] += zmkcs * sinnv + zmkcs_n * cosnvn;
m_coeff_bar_out.lmnsc[mn] += lmksc * cosnv + lmksc_n * sinnvn;
m_coeff_bar_out.lmncs[mn] += lmkcs * sinnv + lmkcs_n * cosnvn;
}
}
}
}
}

#ifdef VMECPP_ENABLE_ENZYME
void IdealMhdModel::applyExactForceJacobianTranspose(
const double* geomP, int geom_stride, FourierForces& m_decomposed_in,
FourierForces& m_physical_f, FourierGeometry& m_physical_scratch,
FourierGeometry& m_decomposed_out, bool fix_m1_gauge) {
const int gS = geom_stride;
const int nForce = (r_.nsMaxFIncludingLcfs - r_.nsMinF) * s_.nZnT;

if (fix_m1_gauge) {
m_decomposed_in.zeroZForceForM1();
}
m_decomposed_in.m1Constraint(1.0 / std::numbers::sqrt2, signOfJacobian);
m_decomposed_in.decomposeInto(m_physical_f, m_p_.scalxc);
if (s_.lthreed) {
dft_ForcesToFourierTranspose_3d_symm(m_physical_f);
} else {
dft_ForcesToFourierTranspose_2d_symm(m_physical_f);
}

std::vector<double> force_bar(kLocalForceBlocks * nForce, 0.0);
auto gather = [&](int b, const Eigen::VectorXd& src) {
const int sz = std::min(nForce, static_cast<int>(src.size()));
for (int i = 0; i < sz; ++i) force_bar[b * nForce + i] = src[i];
};
gather(0, armn_e);
gather(1, armn_o);
gather(2, azmn_e);
gather(3, azmn_o);
gather(4, brmn_e);
gather(5, brmn_o);
gather(6, bzmn_e);
gather(7, bzmn_o);
gather(12, blmn_e);
gather(13, blmn_o);
gather(16, frcon_e);
gather(17, frcon_o);
gather(18, fzcon_e);
gather(19, fzcon_o);
if (s_.lthreed) {
gather(8, crmn_e);
gather(9, crmn_o);
gather(10, czmn_e);
gather(11, czmn_o);
gather(14, clmn_e);
gather(15, clmn_o);
}

std::vector<double> geom_bar(20 * gS, 0.0);
exactForceDensityCotangent(geomP, force_bar.data(), gS, geom_bar.data());

auto scat = [&](int b, Eigen::VectorXd& dst) {
const int sz = std::min(gS, static_cast<int>(dst.size()));
for (int i = 0; i < sz; ++i) dst[i] = geom_bar[b * gS + i];
};
scat(0, r1_e);
scat(1, r1_o);
scat(2, z1_e);
scat(3, z1_o);
scat(4, ru_e);
scat(5, ru_o);
scat(6, zu_e);
scat(7, zu_o);
for (int i = 0; i < gS; ++i) {
lu_e[i] = constants_.lamscale * geom_bar[12 * gS + i];
lu_o[i] = constants_.lamscale * geom_bar[13 * gS + i];
}
if (s_.lthreed) {
scat(8, rv_e);
scat(9, rv_o);
scat(10, zv_e);
scat(11, zv_o);
for (int i = 0; i < gS; ++i) {
lv_e[i] = constants_.lamscale * geom_bar[14 * gS + i];
lv_o[i] = constants_.lamscale * geom_bar[15 * gS + i];
}
}
scat(16, rCon);
scat(17, zCon);

for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {
const double sf = m_p_.sqrtSF[jF - r_.nsMinF1];
for (int kl = 0; kl < s_.nZnT; ++kl) {
const int idx_kl1 = (jF - r_.nsMinF1) * s_.nZnT + kl;
const int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;
ru_e[idx_kl1] += geom_bar[18 * gS + idx_kl];
ru_o[idx_kl1] += sf * geom_bar[18 * gS + idx_kl];
zu_e[idx_kl1] += geom_bar[19 * gS + idx_kl];
zu_o[idx_kl1] += sf * geom_bar[19 * gS + idx_kl];
}
}
if (s_.lthreed) {
dft_FourierToRealTranspose_3d_symm(m_physical_scratch);
} else {
dft_FourierToRealTranspose_2d_symm(m_physical_scratch);
}
m_physical_scratch.extrapolateTowardsAxisTranspose();
m_physical_scratch.m1Constraint(1.0, signOfJacobian);
m_physical_scratch.decomposeInto(m_decomposed_out, m_p_.scalxc);
}

void IdealMhdModel::chipStateVjp(const double* geomP, int geom_stride,
const double* chip_bar,
FourierGeometry& m_physical_scratch,
FourierGeometry& m_decomposed_out) {
const int gS = geom_stride;
const int nForce = (r_.nsMaxFIncludingLcfs - r_.nsMinF) * s_.nZnT;
const int nH = r_.nsMaxH - r_.nsMinH;

std::vector<double> force_bar(kLocalForceBlocks * nForce, 0.0);
for (int jH = 0; jH < nH; ++jH) {
force_bar[20 * nForce + jH] = chip_bar[jH];
}

std::vector<double> geom_bar(20 * gS, 0.0);
exactForceDensityCotangent(geomP, force_bar.data(), gS, geom_bar.data());

auto scat = [&](int b, Eigen::VectorXd& dst) {
const int sz = std::min(gS, static_cast<int>(dst.size()));
for (int i = 0; i < sz; ++i) dst[i] = geom_bar[b * gS + i];
};
scat(0, r1_e);
scat(1, r1_o);
scat(2, z1_e);
scat(3, z1_o);
scat(4, ru_e);
scat(5, ru_o);
scat(6, zu_e);
scat(7, zu_o);
for (int i = 0; i < gS; ++i) {
lu_e[i] = constants_.lamscale * geom_bar[12 * gS + i];
lu_o[i] = constants_.lamscale * geom_bar[13 * gS + i];
}
if (s_.lthreed) {
scat(8, rv_e);
scat(9, rv_o);
scat(10, zv_e);
scat(11, zv_o);
for (int i = 0; i < gS; ++i) {
lv_e[i] = constants_.lamscale * geom_bar[14 * gS + i];
lv_o[i] = constants_.lamscale * geom_bar[15 * gS + i];
}
}

rCon.setZero();
zCon.setZero();
if (s_.lthreed) {
dft_FourierToRealTranspose_3d_symm(m_physical_scratch);
} else {
dft_FourierToRealTranspose_2d_symm(m_physical_scratch);
}
m_physical_scratch.extrapolateTowardsAxisTranspose();
m_physical_scratch.m1Constraint(1.0, signOfJacobian);
m_physical_scratch.decomposeInto(m_decomposed_out, m_p_.scalxc);
}

double IdealMhdModel::composedForceResidual(const double* geomP,
int geom_stride) {
LocalForceComposition comp = makeLocalForceComposition(geom_stride);
const int nForce = comp.force_stride;
std::vector<double> work(LocalForceWorkSize(comp), 0.0);
std::vector<double> force(kLocalForceBlocks * nForce, 0.0);
ComputeLocalForceDensity(geomP, work.data(), force.data(), &comp);

double maxd = 0.0;
auto cmp = [&](int block, const Eigen::VectorXd& prod) {
for (int i = 0; i < static_cast<int>(prod.size()); ++i) {
maxd = std::max(maxd, std::fabs(force[block * nForce + i] - prod[i]));
}
};
cmp(0, armn_e);
cmp(1, armn_o);
cmp(2, azmn_e);
cmp(3, azmn_o);
cmp(4, brmn_e);
cmp(5, brmn_o);
cmp(6, bzmn_e);
cmp(7, bzmn_o);
cmp(12, blmn_e);
cmp(13, blmn_o);
return maxd;
}
#endif

void IdealMhdModel::forcesToFourier(FourierForces& m_physical_f) {
if (s_.lasym) {

symforce();
}

if (s_.lthreed) {
dft_ForcesToFourier_3d_symm(m_physical_f);
} else {
dft_ForcesToFourier_2d_symm(m_physical_f);
}

if (s_.lasym) {
if (s_.lthreed) {
dft_ForcesToFourier_3d_asymm(m_physical_f);
} else {
dft_ForcesToFourier_2d_asymm(m_physical_f);
}
}

if (lforbal) {

const double c = t_.nscale[0];
for (int jF = r_.nsMinFi; jF < r_.nsMaxFi; ++jF) {
if (jF == 0 || jF >= m_fc_.ns - 1) {
continue;
}
const int i = jF - r_.nsMinF;
const int idx_mn = ((jF - r_.nsMinF) * s_.mpol + 1) * (s_.ntor + 1);
const double equif = m_p_.equiF[jF - r_.nsMinFi];
const double frcc = m_physical_f.frcc[idx_mn];
const double fzsc = m_physical_f.fzsc[idx_mn];
const double work1 = frcc_fac[i] * frcc + fzsc_fac[i] * fzsc;
m_physical_f.frcc[idx_mn] = rzu_fac[i] * (c * equif + work1);
m_physical_f.fzsc[idx_mn] = rru_fac[i] * (c * equif - work1);
}
}
}

void IdealMhdModel::dft_ForcesToFourier_3d_symm(FourierForces& m_physical_f) {
const auto input_data = RealSpaceForces{
.armn_e = armn_e,
.armn_o = armn_o,
.azmn_e = azmn_e,
.azmn_o = azmn_o,
.blmn_e = blmn_e,
.blmn_o = blmn_o,
.brmn_e = brmn_e,
.brmn_o = brmn_o,
.bzmn_e = bzmn_e,
.bzmn_o = bzmn_o,
.clmn_e = clmn_e,
.clmn_o = clmn_o,
.crmn_e = crmn_e,
.crmn_o = crmn_o,
.czmn_e = czmn_e,
.czmn_o = czmn_o,
.frcon_e = frcon_e,
.frcon_o = frcon_o,
.fzcon_e = fzcon_e,
.fzcon_o = fzcon_o,
};

ForcesToFourier3DSymmFastPoloidal(input_data, xmpq, r_, m_fc_, s_, t_,
m_vacuum_pressure_state_, m_physical_f);
}

void IdealMhdModel::dft_ForcesToFourier_3d_asymm(FourierForces& m_physical_f) {
const auto input_data = RealSpaceForces{
.armn_e = armn_asym_e,
.armn_o = armn_asym_o,
.azmn_e = azmn_asym_e,
.azmn_o = azmn_asym_o,
.blmn_e = blmn_asym_e,
.blmn_o = blmn_asym_o,
.brmn_e = brmn_asym_e,
.brmn_o = brmn_asym_o,
.bzmn_e = bzmn_asym_e,
.bzmn_o = bzmn_asym_o,
.clmn_e = clmn_asym_e,
.clmn_o = clmn_asym_o,
.crmn_e = crmn_asym_e,
.crmn_o = crmn_asym_o,
.czmn_e = czmn_asym_e,
.czmn_o = czmn_asym_o,
.frcon_e = frcon_asym_e,
.frcon_o = frcon_asym_o,
.fzcon_e = fzcon_asym_e,
.fzcon_o = fzcon_asym_o,
};
ForcesToFourier3DAsymFastPoloidal(input_data, xmpq, r_, m_fc_, s_, t_,
m_vacuum_pressure_state_, m_physical_f);
}

void IdealMhdModel::dft_ForcesToFourier_2d_symm(FourierForces& m_physical_f) {

m_physical_f.setZero();

#ifdef _OPENMP
#pragma omp barrier
#endif

int jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns - 1);
if (m_fc_.lfreeb &&
m_vacuum_pressure_state_ >= VacuumPressureState::kInitialized) {

jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns);
}

for (int jF = r_.nsMinF; jF < jMaxRZ; ++jF) {

int num_m = s_.mpol;
if (jF == 0) {

num_m = 1;
}

for (int m = 0; m < num_m; ++m) {
const bool m_even = m % 2 == 0;
const int idx_jm = (jF - r_.nsMinF) * s_.mpol + m;

const auto& armn = m_even ? armn_e : armn_o;
const auto& brmn = m_even ? brmn_e : brmn_o;
const auto& azmn = m_even ? azmn_e : azmn_o;
const auto& bzmn = m_even ? bzmn_e : bzmn_o;
const auto& frcon = m_even ? frcon_e : frcon_o;
const auto& fzcon = m_even ? fzcon_e : fzcon_o;

for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_jl =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;

const double rnkcc = armn[idx_jl];
const double rnkcc_m = brmn[idx_jl];
const double znksc = azmn[idx_jl];
const double znksc_m = bzmn[idx_jl];

const double rcon_cc = frcon[idx_jl];
const double zcon_sc = fzcon[idx_jl];

const int idx_ml = m * s_.nThetaReduced + l;
const double cosmui = t_.cosmui[idx_ml];
const double sinmumi = t_.sinmumi[idx_ml];
const double sinmui = t_.sinmui[idx_ml];
const double cosmumi = t_.cosmumi[idx_ml];

const double _rcc = rnkcc + xmpq[m] * rcon_cc;
m_physical_f.frcc[idx_jm] += _rcc * cosmui + rnkcc_m * sinmumi;

const double _zsc = znksc + xmpq[m] * zcon_sc;
m_physical_f.fzsc[idx_jm] += _zsc * sinmui + znksc_m * cosmumi;
}
}
}
}

for (int jF = std::max(1, r_.nsMinF); jF < r_.nsMaxFIncludingLcfs; ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
const int m_even = m % 2 == 0;
const auto& blmn = m_even ? blmn_e : blmn_o;
const int idx_jm = (jF - r_.nsMinF) * s_.mpol + m;

for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_jl =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
const double lnksc_m = blmn[idx_jl];

const double cosmumi = t_.cosmumi[m * s_.nThetaReduced + l];
m_physical_f.flsc[idx_jm] += lnksc_m * cosmumi;
}
}
}
}
}

void IdealMhdModel::symforce() {
const int nThetaEff = s_.nThetaEff;
const bool lthreed = s_.lthreed;

auto fold = [&](Eigen::VectorXd& f, Eigen::VectorXd& fa, int jLo, int jHi,
bool std_parity) {
for (int jF = jLo; jF < jHi; ++jF) {
const int base = (jF - r_.nsMinF) * s_.nZnT;
for (int k = 0; k < s_.nZeta; ++k) {
const int kRev = (s_.nZeta - k) % s_.nZeta;
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int jl = base + k * nThetaEff + l;
const int jlRev =
base + kRev * nThetaEff + ((s_.nThetaEven - l) % s_.nThetaEven);
fa[jl] =
std_parity ? 0.5 * (f[jl] - f[jlRev]) : 0.5 * (f[jl] + f[jlRev]);
}
}
for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int jl = base + k * nThetaEff + l;
f[jl] -= fa[jl];
}
}
}
};

const int jRZ = r_.nsMaxF;
const int jL = r_.nsMaxFIncludingLcfs;

fold(armn_e, armn_asym_e, r_.nsMinF, jRZ,  true);
fold(armn_o, armn_asym_o, r_.nsMinF, jRZ, true);
fold(brmn_e, brmn_asym_e, r_.nsMinF, jRZ, false);
fold(brmn_o, brmn_asym_o, r_.nsMinF, jRZ, false);
fold(azmn_e, azmn_asym_e, r_.nsMinF, jRZ, false);
fold(azmn_o, azmn_asym_o, r_.nsMinF, jRZ, false);
fold(bzmn_e, bzmn_asym_e, r_.nsMinF, jRZ, true);
fold(bzmn_o, bzmn_asym_o, r_.nsMinF, jRZ, true);
fold(frcon_e, frcon_asym_e, r_.nsMinF, jRZ, true);
fold(frcon_o, frcon_asym_o, r_.nsMinF, jRZ, true);
fold(fzcon_e, fzcon_asym_e, r_.nsMinF, jRZ, false);
fold(fzcon_o, fzcon_asym_o, r_.nsMinF, jRZ, false);

fold(blmn_e, blmn_asym_e, r_.nsMinF, jL, true);
fold(blmn_o, blmn_asym_o, r_.nsMinF, jL, true);

if (lthreed) {
fold(crmn_e, crmn_asym_e, r_.nsMinF, jRZ, false);
fold(crmn_o, crmn_asym_o, r_.nsMinF, jRZ, false);
fold(czmn_e, czmn_asym_e, r_.nsMinF, jRZ, true);
fold(czmn_o, czmn_asym_o, r_.nsMinF, jRZ, true);
fold(clmn_e, clmn_asym_e, r_.nsMinF, jL, true);
fold(clmn_o, clmn_asym_o, r_.nsMinF, jL, true);
}
}

void IdealMhdModel::dft_ForcesToFourier_2d_asymm(FourierForces& m_physical_f) {
int jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns - 1);
if (m_fc_.lfreeb &&
m_vacuum_pressure_state_ >= VacuumPressureState::kInitialized) {
jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns);
}

for (int jF = r_.nsMinF; jF < jMaxRZ; ++jF) {
int num_m = s_.mpol;
if (jF == 0) {
num_m = 1;
}

for (int m = 0; m < num_m; ++m) {
const bool m_even = m % 2 == 0;
const int idx_jm = (jF - r_.nsMinF) * s_.mpol + m;

const auto& armn = m_even ? armn_asym_e : armn_asym_o;
const auto& brmn = m_even ? brmn_asym_e : brmn_asym_o;
const auto& azmn = m_even ? azmn_asym_e : azmn_asym_o;
const auto& bzmn = m_even ? bzmn_asym_e : bzmn_asym_o;
const auto& frcon = m_even ? frcon_asym_e : frcon_asym_o;
const auto& fzcon = m_even ? fzcon_asym_e : fzcon_asym_o;

for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_jl =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;

const double rnksc = armn[idx_jl];
const double rnksc_m = brmn[idx_jl];
const double znkcc = azmn[idx_jl];
const double znkcc_m = bzmn[idx_jl];
const double rcon_sc = frcon[idx_jl];
const double zcon_cc = fzcon[idx_jl];

const int idx_ml = m * s_.nThetaReduced + l;
const double cosmui = t_.cosmui[idx_ml];
const double sinmui = t_.sinmui[idx_ml];
const double cosmumi = t_.cosmumi[idx_ml];
const double sinmumi = t_.sinmumi[idx_ml];

const double _rsc = rnksc + xmpq[m] * rcon_sc;
m_physical_f.frsc[idx_jm] += _rsc * sinmui + rnksc_m * cosmumi;

const double _zcc = znkcc + xmpq[m] * zcon_cc;
m_physical_f.fzcc[idx_jm] += _zcc * cosmui + znkcc_m * sinmumi;
}
}
}
}

for (int jF = std::max(1, r_.nsMinF); jF < r_.nsMaxFIncludingLcfs; ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
const int m_even = m % 2 == 0;
const auto& blmn = m_even ? blmn_asym_e : blmn_asym_o;
const int idx_jm = (jF - r_.nsMinF) * s_.mpol + m;

for (int k = 0; k < s_.nZeta; ++k) {
for (int l = 0; l < s_.nThetaReduced; ++l) {
const int idx_jl =
((jF - r_.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
const double lnkcc_m = blmn[idx_jl];

const double sinmumi = t_.sinmumi[m * s_.nThetaReduced + l];
m_physical_f.flcc[idx_jm] += lnkcc_m * sinmumi;
}
}
}
}
}

void IdealMhdModel::applyM1Preconditioner(FourierForces& m_decomposed_f) {
if (!s_.lthreed && !s_.lasym) {

return;
}

for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
for (int n = 0; n < s_.ntor + 1; ++n) {
int m = 1;
int mPar = m % 2;

double denom =
ard[(jF - r_.nsMinF) * 2 + mPar] + brd[(jF - r_.nsMinF) * 2 + mPar] +
azd[(jF - r_.nsMinF) * 2 + mPar] + bzd[(jF - r_.nsMinF) * 2 + mPar];
double forceScaleR = (ard[(jF - r_.nsMinF) * 2 + mPar] +
brd[(jF - r_.nsMinF) * 2 + mPar]) /
denom;
double forceScaleZ = (azd[(jF - r_.nsMinF) * 2 + mPar] +
bzd[(jF - r_.nsMinF) * 2 + mPar]) /
denom;

int idx_mn = ((jF - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;

if (s_.lthreed) {
m_decomposed_f.frss[idx_mn] *= forceScaleR;
m_decomposed_f.fzcs[idx_mn] *= forceScaleZ;
}
if (s_.lasym) {
m_decomposed_f.frsc[idx_mn] *= forceScaleR;
m_decomposed_f.fzcc[idx_mn] *= forceScaleZ;
}
}
}

#ifdef _OPENMP
#pragma omp barrier
#endif
}

void IdealMhdModel::assembleRZPreconditioner() {
for (int m = 0; m < s_.mpol; ++m) {

int jMin = 0;
if (m > 0) {
jMin = 1;
}

for (int n = 0; n < s_.ntor + 1; ++n) {
int mn = m * (s_.ntor + 1) + n;
this->jMin[mn] = jMin;
}
}

int jMax = m_fc_.ns - 1;
if (m_fc_.lfreeb &&
m_vacuum_pressure_state_ >= VacuumPressureState::kInitialized) {
jMax = m_fc_.ns;
}

for (int jF = r_.nsMinF; jF < std::min(r_.nsMaxF, jMax); ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
const int m_parity = m % 2;
for (int n = 0; n < s_.ntor + 1; ++n) {
int mn = m * (s_.ntor + 1) + n;
int idx_mn = ((jF - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;
if (jF >= jMin[mn]) {

if (jF < r_.nsMaxH) {
ar[idx_mn] = -(arm[(jF - r_.nsMinH) * 2 + m_parity] +
brm[(jF - r_.nsMinH) * 2 + m_parity] * m * m);
az[idx_mn] = -(azm[(jF - r_.nsMinH) * 2 + m_parity] +
bzm[(jF - r_.nsMinH) * 2 + m_parity] * m * m);
}

dr[idx_mn] = -(ard[(jF - r_.nsMinF) * 2 + m_parity] +
brd[(jF - r_.nsMinF) * 2 + m_parity] * m * m +
cxd[jF - r_.nsMinF] * n * s_.nfp * n * s_.nfp);
dz[idx_mn] = -(azd[(jF - r_.nsMinF) * 2 + m_parity] +
bzd[(jF - r_.nsMinF) * 2 + m_parity] * m * m +
cxd[jF - r_.nsMinF] * n * s_.nfp * n * s_.nfp);

if (jF > 0) {
br[idx_mn] = -(arm[(jF - 1 - r_.nsMinH) * 2 + m_parity] +
brm[(jF - 1 - r_.nsMinH) * 2 + m_parity] * m * m);
bz[idx_mn] = -(azm[(jF - 1 - r_.nsMinH) * 2 + m_parity] +
bzm[(jF - 1 - r_.nsMinH) * 2 + m_parity] * m * m);
}

if (jF == 1 && m == 1) {

dr[idx_mn] += br[idx_mn];
dz[idx_mn] += bz[idx_mn];
}
} else {
ar[idx_mn] = 0.0;
az[idx_mn] = 0.0;
dr[idx_mn] = 0.0;
dz[idx_mn] = 0.0;
br[idx_mn] = 0.0;
bz[idx_mn] = 0.0;
}
}
}
}

if (r_.nsMaxF == m_fc_.ns) {

const double edge_pedestal = 0.05;
for (int n = 0; n < s_.ntor + 1; ++n) {
{
int m = 0;
int idx_mn =
((m_fc_.ns - 1 - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;
dr[idx_mn] *= 1.0 + edge_pedestal;
dz[idx_mn] *= 1.0 + edge_pedestal;
}
{
int m = 1;
int idx_mn =
((m_fc_.ns - 1 - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;
dr[idx_mn] *= 1.0 + edge_pedestal;
dz[idx_mn] *= 1.0 + edge_pedestal;
}
for (int m = 2; m < s_.mpol; ++m) {
int idx_mn =
((m_fc_.ns - 1 - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;
dr[idx_mn] *= 1.0 + 2.0 * edge_pedestal;
dz[idx_mn] *= 1.0 + 2.0 * edge_pedestal;
}
}

double fac = 0.25;
double multFact = std::min(fac, fac * m_fc_.deltaS * 15.0);

int idx_00 = (m_fc_.ns - 1 - r_.nsMinF) * s_.mpol * (s_.ntor + 1);
dz[idx_00] *= (1.0 - multFact) / (1.0 + edge_pedestal);
}

#ifdef _OPENMP
#pragma omp barrier
#endif
}

absl::Status IdealMhdModel::applyRZPreconditioner(
FourierForces& m_decomposed_f) {

std::array<std::span<double>, 4> cR{};
std::array<std::span<double>, 4> cZ{};
{
int idx_basis = 0;

cR[idx_basis] = m_decomposed_f.frcc;
cZ[idx_basis] = m_decomposed_f.fzsc;
idx_basis++;
if (s_.lthreed) {
cR[idx_basis] = m_decomposed_f.frss;
cZ[idx_basis] = m_decomposed_f.fzcs;
idx_basis++;
}
if (s_.lasym) {
cR[idx_basis] = m_decomposed_f.frsc;
cZ[idx_basis] = m_decomposed_f.fzcc;
idx_basis++;
if (s_.lthreed) {
cR[idx_basis] = m_decomposed_f.frcs;
cZ[idx_basis] = m_decomposed_f.fzss;
idx_basis++;
}
}

if (idx_basis != s_.num_basis) {
return absl::InternalError(
absl::StrFormat("counting error: idx_basis=%d != num_basis=%d",
idx_basis, s_.num_basis));
}
}

int jMax = m_fc_.ns - 1;
if (m_fc_.lfreeb &&
m_vacuum_pressure_state_ >= VacuumPressureState::kInitialized) {
jMax = m_fc_.ns;
}

for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (jF - r_.nsMinF) * s_.mnsize + mn;
m_h_.all_ar(mn, jF) = ar[idx_mn];
m_h_.all_az(mn, jF) = az[idx_mn];
m_h_.all_dr(mn, jF) = dr[idx_mn];
m_h_.all_dz(mn, jF) = dz[idx_mn];
m_h_.all_br(mn, jF) = br[idx_mn];
m_h_.all_bz(mn, jF) = bz[idx_mn];
for (int idx_basis = 0; idx_basis < s_.num_basis; ++idx_basis) {
m_h_.all_cr[mn](idx_basis, jF) = cR[idx_basis][idx_mn];
m_h_.all_cz[mn](idx_basis, jF) = cZ[idx_basis][idx_mn];
}
}
}
#ifdef _OPENMP
#pragma omp barrier
#endif

const int thread_id = r_.get_thread_id();
const int num_threads = r_.get_num_threads();
const int mnstep = s_.mnsize / num_threads;
const int remainder = s_.mnsize % num_threads;
int mnmin = thread_id * mnstep;
int mnmax = mnmin + mnstep;

if (thread_id < remainder) {
mnmin += thread_id;
mnmax += thread_id + 1;
} else {
mnmin += remainder;
mnmax += remainder;
}

const int ns = m_h_.all_ar.cols();
for (int mn = mnmin; mn < mnmax; ++mn) {
TridiagonalSolveSerial(std::span<double>(m_h_.all_ar.row(mn).data(), ns),
std::span<double>(m_h_.all_dr.row(mn).data(), ns),
std::span<double>(m_h_.all_br.row(mn).data(), ns),
m_h_.all_cr[mn].data(), ns, jMin[mn], jMax,
s_.num_basis);
TridiagonalSolveSerial(std::span<double>(m_h_.all_az.row(mn).data(), ns),
std::span<double>(m_h_.all_dz.row(mn).data(), ns),
std::span<double>(m_h_.all_bz.row(mn).data(), ns),
m_h_.all_cz[mn].data(), ns, jMin[mn], jMax,
s_.num_basis);
}
#ifdef _OPENMP
#pragma omp barrier
#endif

for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (jF - r_.nsMinF) * s_.mnsize + mn;
for (int idx_basis = 0; idx_basis < s_.num_basis; ++idx_basis) {
cR[idx_basis][idx_mn] = m_h_.all_cr[mn](idx_basis, jF);
cZ[idx_basis][idx_mn] = m_h_.all_cz[mn](idx_basis, jF);
}
}
}

return absl::OkStatus();
}

void IdealMhdModel::applyLambdaPreconditioner(FourierForces& m_decomposed_f) {
for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {
for (int m = 0; m < s_.mpol; ++m) {
for (int n = 0; n <= s_.ntor; ++n) {
int idx_mn = ((jF - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;

m_decomposed_f.flsc[idx_mn] *= lambdaPreconditioner[idx_mn];
if (s_.lthreed) {
m_decomposed_f.flcs[idx_mn] *= lambdaPreconditioner[idx_mn];
}
if (s_.lasym) {
m_decomposed_f.flcc[idx_mn] *= lambdaPreconditioner[idx_mn];
if (s_.lthreed) {
m_decomposed_f.flss[idx_mn] *= lambdaPreconditioner[idx_mn];
}
}
}
}
}
}

double IdealMhdModel::get_delbsq() const {
double delBSqAvg = 0.0;
if (m_fc_.lfreeb &&
m_vacuum_pressure_state_ >= VacuumPressureState::kActive) {
double delBSqNorm = 0.0;
for (int kl = 0; kl < s_.nZnT; ++kl) {
int l = kl % s_.nThetaEff;
delBSqAvg += delBSq[kl] * s_.wInt[l];
delBSqNorm += insideTotalPressure[kl] * s_.wInt[l];
}
delBSqAvg /= delBSqNorm;
}
return delBSqAvg;
}

int IdealMhdModel::get_ivacskip() const { return ivacskip; }

}

// source: vmecpp/vmec/iteration_logger/iteration_logger.cc

// header: vmecpp/vmec/iteration_logger/iteration_logger.h

#ifndef VMECPP_VMEC_ITERATION_LOGGER_ITERATION_LOGGER_H_
#define VMECPP_VMEC_ITERATION_LOGGER_ITERATION_LOGGER_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace vmecpp {

enum class OutputMode : std::uint8_t {
kSilent = 0,
kLegacy = 1,
kProgress = 2,
kProgressNonTTY = 3
};

struct RunSummary {
bool converged = false;
int total_iterations = 0;
int num_jacobian_resets = 0;
double fsqr = 0.0;
double fsqz = 0.0;
double fsql = 0.0;
double ftolv = 0.0;
double betatot = 0.0;
double betapol = 0.0;
double betator = 0.0;
double w_mhd = 0.0;
double rax = 0.0;
double aminor = 0.0;
double rmajor = 0.0;
double b0 = 0.0;
};

class IterationLogger {
public:
IterationLogger(std::ostream& output, OutputMode mode);

void BeginStage(int stage_index, int num_stages, int ns, int mnmax,
double ftolv, int niter, bool is_free_boundary);

void LogIteration(int iter, double fsqr, double fsqz, double fsql,
double fsqr1, double fsqz1, double fsql1, double delt,
double rax, double w_mhd, double beta_vol_avg,
double vol_avg_m, double delbsq);

void EndStage(double mhd_energy);

void EndRun(const RunSummary& summary);

private:

struct StageState {
int ns = 0;
int niter = 0;
int last_iter = 0;
double ftolv = 0.0;
double initial_log_fsq = 0.0;
double last_fsq = 0.0;
double last_beta = 0.0;
double last_w_mhd = 0.0;
double last_rax = 0.0;
double last_delbsq = 0.0;
bool initial_fsq_set = false;
bool completed = false;
};

double ComputeProgress(const StageState& stage) const;

void RenderProgressDisplay();

void RenderSingleLineProgress();

std::string FormatBar(double fraction, int width,
const char* fill_color = nullptr) const;

void PrintSummaryTable(const RunSummary& summary) const;

void PrintLegacyHeader() const;

void PrintLegacyRow(int iter, double fsqr, double fsqz, double fsql,
double fsqr1, double fsqz1, double fsql1, double delt,
double rax, double w_mhd, double beta_vol_avg,
double vol_avg_m, double delbsq) const;

std::ostream& output_;
OutputMode mode_;
bool is_free_boundary_ = false;
int current_stage_ = -1;
int num_stages_ = 0;
std::vector<StageState> stages_;

int progress_lines_printed_ = 0;
};

}

#endif

#include <algorithm>
#include <cmath>

#include "absl/strings/str_format.h"

namespace vmecpp {

namespace {

constexpr const char* kBold = "\033[1m";
constexpr const char* kDim = "\033[2m";
constexpr const char* kReset = "\033[0m";
constexpr const char* kGreen = "\033[32m";
constexpr const char* kYellow = "\033[33m";
constexpr const char* kRed = "\033[31m";

std::string CursorUp(int n) {
if (n <= 0) return "";
return absl::StrFormat("\033[%dA", n);
}

constexpr const char* kClearLine = "\033[K";

}

IterationLogger::IterationLogger(std::ostream& output, OutputMode mode)
: output_(output), mode_(mode) {}

void IterationLogger::BeginStage(int stage_index, int num_stages, int ns,
int mnmax, double ftolv, int niter,
bool is_free_boundary) {
if (mode_ == OutputMode::kSilent) return;

current_stage_ = stage_index;
num_stages_ = num_stages;
is_free_boundary_ = is_free_boundary;

if (stage_index >= static_cast<int>(stages_.size())) {
stages_.resize(stage_index + 1);
}

StageState& stage = stages_[stage_index];
stage.ns = ns;
stage.niter = niter;
stage.ftolv = ftolv;
stage.last_iter = 0;
stage.initial_fsq_set = false;
stage.completed = false;

if (mode_ == OutputMode::kLegacy) {
output_ << absl::StrFormat(
"\n NS = %d   NO. FOURIER MODES = %d   FTOLV = %9.3e   NITER = %d\n",
ns, mnmax, ftolv, niter);
PrintLegacyHeader();
}

}

void IterationLogger::LogIteration(int iter, double fsqr, double fsqz,
double fsql, double fsqr1, double fsqz1,
double fsql1, double delt, double rax,
double w_mhd, double beta_vol_avg,
double vol_avg_m, double delbsq) {
if (mode_ == OutputMode::kSilent) return;

if (mode_ == OutputMode::kLegacy) {
PrintLegacyRow(iter, fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1, delt, rax,
w_mhd, beta_vol_avg, vol_avg_m, delbsq);
return;
}

if (current_stage_ < 0 ||
current_stage_ >= static_cast<int>(stages_.size())) {
return;
}

StageState& stage = stages_[current_stage_];
double fsq_total = fsqr + fsqz + fsql;

if (!stage.initial_fsq_set && fsq_total > 0.0) {
stage.initial_log_fsq = std::log10(fsq_total);
stage.initial_fsq_set = true;
}

stage.last_iter = iter;
stage.last_fsq = fsq_total;
stage.last_beta = beta_vol_avg;
stage.last_w_mhd = w_mhd;
stage.last_rax = rax;
stage.last_delbsq = delbsq;

if (mode_ == OutputMode::kProgress) {
RenderProgressDisplay();
} else {
RenderSingleLineProgress();
}
}

void IterationLogger::EndStage(double mhd_energy) {
if (mode_ == OutputMode::kSilent) return;

if (current_stage_ >= 0 &&
current_stage_ < static_cast<int>(stages_.size())) {
stages_[current_stage_].completed = true;
}

if (mode_ == OutputMode::kLegacy) {
output_ << absl::StrFormat("MHD Energy = %12.6e\n", mhd_energy);
output_ << std::flush;
return;
}

if (mode_ == OutputMode::kProgress) {
RenderProgressDisplay();
} else {

RenderSingleLineProgress();
output_ << "\n";
}
}

void IterationLogger::EndRun(const RunSummary& summary) {
if (mode_ == OutputMode::kSilent || mode_ == OutputMode::kLegacy) return;

PrintSummaryTable(summary);
output_ << std::flush;
}

double IterationLogger::ComputeProgress(const StageState& stage) const {
if (stage.completed) return 1.0;

double fsq_progress = 0.0;
if (stage.initial_fsq_set && stage.last_fsq > 0.0) {
double log_current = std::log10(stage.last_fsq);
double log_target = std::log10(stage.ftolv);
if (stage.initial_log_fsq > log_target) {
fsq_progress = (stage.initial_log_fsq - log_current) /
(stage.initial_log_fsq - log_target);
}
}

double iter_progress = 0.0;
if (stage.niter > 0) {
iter_progress =
static_cast<double>(stage.last_iter) / static_cast<double>(stage.niter);
}

double progress = std::max(fsq_progress, iter_progress);
return std::max(0.0, std::min(1.0, progress));
}

void IterationLogger::RenderProgressDisplay() {

if (progress_lines_printed_ > 0) {
output_ << CursorUp(progress_lines_printed_);
}

int lines = 0;

output_ << kClearLine;
for (int i = 0; i < static_cast<int>(stages_.size()); ++i) {
if (i > 0) output_ << "  ";
const StageState& s = stages_[i];

if (s.completed) {
output_ << kGreen << kBold;
} else if (i == current_stage_) {
output_ << kBold;
} else {
output_ << kDim;
}

output_ << absl::StrFormat("[%d/%d] NS=%d", i + 1, num_stages_, s.ns);
output_ << kReset;
}
output_ << "\n";
lines++;

output_ << kClearLine << "\n";
lines++;

for (int i = 0; i < static_cast<int>(stages_.size()); ++i) {
const StageState& s = stages_[i];

if (!s.initial_fsq_set && !s.completed) {

output_ << kClearLine;
output_ << absl::StrFormat("  Stage %d (NS=%3d): ", i + 1, s.ns);
output_ << "[" << kDim << std::string(50, '-') << kReset << "]" << kDim
<< "   0.0%" << kReset << "\n";
lines++;

output_ << kClearLine << "  " << kDim << "waiting..." << kReset << "\n";
lines++;

output_ << kClearLine << "\n";
lines++;
continue;
}

double progress_frac = ComputeProgress(s);

const char* fill_color = kRed;
if (progress_frac >= 1.0) {
fill_color = kGreen;
} else if (progress_frac > 0.5) {
fill_color = kYellow;
}

output_ << kClearLine;
output_ << absl::StrFormat("  Stage %d (NS=%3d): ", i + 1, s.ns);
output_ << FormatBar(progress_frac, 50, fill_color);
output_ << absl::StrFormat(" %5.1f%%", progress_frac * 100.0);
output_ << "\n";
lines++;

output_ << kClearLine << "  " << kDim;
output_ << absl::StrFormat(
"iter=%-5d FSQ=%.2e  beta=%.4f  W_MHD=%.4f  "
"RAX=%.4f",
s.last_iter, s.last_fsq, s.last_beta, s.last_w_mhd, s.last_rax);
if (is_free_boundary_) {
output_ << absl::StrFormat("  DELBSQ=%.3e", s.last_delbsq);
}
output_ << kReset << "\n";
lines++;

output_ << kClearLine << "\n";
lines++;
}

output_ << std::flush;
progress_lines_printed_ = lines;
}

void IterationLogger::RenderSingleLineProgress() {
if (current_stage_ < 0 ||
current_stage_ >= static_cast<int>(stages_.size())) {
return;
}

const StageState& s = stages_[current_stage_];
double progress_frac = ComputeProgress(s);

const char* fill_color = kRed;
if (progress_frac >= 1.0) {
fill_color = kGreen;
} else if (progress_frac > 0.5) {
fill_color = kYellow;
}

output_ << "\r";
output_ << absl::StrFormat("Stage %d/%d [ns=%d] ", current_stage_ + 1,
num_stages_, s.ns);
output_ << FormatBar(progress_frac, 30, fill_color);
output_ << absl::StrFormat(" %5.1f%%  FSQ=%.2e", progress_frac * 100.0,
s.last_fsq);
if (is_free_boundary_) {
output_ << absl::StrFormat("  delbsq=%.2e", s.last_delbsq);
}

output_ << "    ";
output_ << std::flush;
}

std::string IterationLogger::FormatBar(double fraction, int width,
const char* fill_color) const {
fraction = std::max(0.0, std::min(1.0, fraction));
int filled = static_cast<int>(fraction * width);
std::string bar = "[";
if (fill_color != nullptr && filled > 0) {
bar += fill_color;
}
bar.append(filled, '=');
if (fill_color != nullptr && filled > 0) {
bar += kReset;
}
bar.append(width - filled, '-');
bar += "]";
return bar;
}

void IterationLogger::PrintSummaryTable(const RunSummary& summary) const {

const bool use_ansi =
(mode_ == OutputMode::kProgress || mode_ == OutputMode::kProgressNonTTY);
const char* bold = use_ansi ? kBold : "";
const char* green = use_ansi ? kGreen : "";
const char* red = use_ansi ? kRed : "";
const char* reset = use_ansi ? kReset : "";

const char* tl = "\xe2\x95\xad";
const char* tr = "\xe2\x95\xae";
const char* bl = "\xe2\x95\xb0";
const char* br = "\xe2\x95\xaf";
const char* h = "\xe2\x94\x80";
const char* v = "\xe2\x94\x82";

const int table_width = 38;

std::string hline;
for (int i = 0; i < table_width; ++i) hline += h;

std::string top_border = std::string(tl) + hline + tr;
std::string bot_border = std::string(bl) + hline + br;

auto row = [&](const std::string& content) {
output_ << v << content << v << "\n";
};
auto blank_row = [&]() { row(std::string(table_width, ' ')); };

const char* lt = "\xe2\x94\x9c";
const char* rt = "\xe2\x94\xa4";
std::string sep_border = std::string(lt) + hline + rt;

output_ << "\n" << top_border << "\n";
row(absl::StrFormat("          %sVMEC++ Run Summary%14s", bold, reset));
output_ << sep_border << "\n";

if (summary.converged) {
row(absl::StrFormat(" %-15s %s%sCONVERGED%s            ", "Status", green,
bold, reset));
} else {
row(absl::StrFormat(" %-15s %s%sNOT CONVERGED%s        ", "Status", red,
bold, reset));
}

row(absl::StrFormat(" %-15s %-20d ", "Jacobian resets",
summary.num_jacobian_resets));
blank_row();

row(absl::StrFormat(" %-15s %-20.4e ", "Final FSQR", summary.fsqr));
row(absl::StrFormat(" %-15s %-20.4e ", "Final FSQZ", summary.fsqz));
row(absl::StrFormat(" %-15s %-20.4e ", "Final FSQL", summary.fsql));
blank_row();

row(absl::StrFormat(" %-14s %8.4f %%             ", "β total   ",
100 * summary.betatot));
row(absl::StrFormat(" %-14s %8.4f %%             ", "β poloidal",
100 * summary.betapol));
row(absl::StrFormat(" %-14s %8.4f %%             ", "β toroidal",
100 * summary.betator));
blank_row();

row(absl::StrFormat(" %-14s % -20.4f  ", "R_major", summary.rmajor));
row(absl::StrFormat(" %-14s % -20.4f  ", "a_minor", summary.aminor));
row(absl::StrFormat(" %-14s % -20.4f  ", "B0", summary.b0));

output_ << bot_border << "\n";
}

void IterationLogger::PrintLegacyHeader() const {
output_ << '\n';
if (is_free_boundary_) {
output_ << " ITER |    FSQR     FSQZ     FSQL    |    fsqr     fsqz      "
"fsql   |   DELT   |  RAX(v=0) |    W_MHD   |   <BETA>   |  "
"<M>  |  DELBSQ  \n";
output_ << "------+------------------------------+-----------------------"
"-------+----------+-----------+------------+------------+----"
"---+----------\n";
} else {
output_ << " ITER |    FSQR     FSQZ     FSQL    |    fsqr     fsqz    "
"  fsql  "
" |   DELT   |  RAX(v=0) |    W_MHD   |   <BETA>   |  <M>  \n";
output_ << "------+------------------------------+---------------------"
"--------"
"-+----------+-----------+------------+------------+-------\n";
}
}

void IterationLogger::PrintLegacyRow(int iter, double fsqr, double fsqz,
double fsql, double fsqr1, double fsqz1,
double fsql1, double delt, double rax,
double w_mhd, double beta_vol_avg,
double vol_avg_m, double delbsq) const {
if (is_free_boundary_) {
output_ << absl::StrFormat(
"%5d | %.2e  %.2e  %.2e | %.2e  %.2e  %.2e | %.2e | "
"%.3e | %.4e | %.4e | %5.3f | %.3e\n",
iter, fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1, delt, rax, w_mhd,
beta_vol_avg, vol_avg_m, delbsq);
} else {
output_ << absl::StrFormat(
"%5d | %.2e  %.2e  %.2e | %.2e  %.2e  %.2e | %.2e | "
"%.3e | %.4e | %.4e | %5.3f\n",
iter, fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1, delt, rax, w_mhd,
beta_vol_avg, vol_avg_m);
}
}

}

// source: vmecpp/vmec/output_quantities/output_quantities.cc

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "absl/log/check.h"

using Eigen::VectorXd;
using hdf5_io::ReadH5Dataset;
using hdf5_io::WriteH5Dataset;
using testing::IsCloseRelAbs;
using testing::IsVectorCloseRelAbs;

namespace {

VectorXd NonEmptyVectorOr(const Eigen::VectorXd& vec, const double val) {
if (vec.size() > 0) {
return vec;
} else {
return VectorXd::Constant(1, val);
}
}
}

#define WRITEMEMBER(x) \
WriteH5Dataset(x, absl::StrFormat("%s/%s", H5key, #x), file);
#define READMEMBER(x) \
ReadH5Dataset(m_obj.x, absl::StrFormat("%s/%s", H5key, #x), from_file);

#define READMEMBER_COMPAT(x, old_name)                                \
if (from_file.nameExists(absl::StrFormat("%s/%s", H5key, #x))) {    \
READMEMBER(x);                                                    \
} else {                                                            \
ReadH5Dataset(m_obj.x, absl::StrFormat("%s/%s", H5key, old_name), \
from_file);                                         \
}

#define READMEMBER_OPTIONAL(x)                                     \
if (from_file.nameExists(absl::StrFormat("%s/%s", H5key, #x))) { \
READMEMBER(x);                                                 \
}

absl::Status vmecpp::VmecInternalResults::WriteTo(H5::H5File& file) const {
file.createGroup(H5key);
WRITEMEMBER(sign_of_jacobian);
WRITEMEMBER(lamscale);
WRITEMEMBER(num_full);
WRITEMEMBER(num_half);
WRITEMEMBER(nZnT_reduced);
WRITEMEMBER(sqrtSH);
WRITEMEMBER(sqrtSF);
WRITEMEMBER(sm);
WRITEMEMBER(sp);
WRITEMEMBER(phipF);
WRITEMEMBER(chipF);
WRITEMEMBER(phipH);
WRITEMEMBER(chipH);
WRITEMEMBER(currH);
WRITEMEMBER(phiF);
WRITEMEMBER(iotaF);
WRITEMEMBER(spectral_width);
WRITEMEMBER(bvcoH);
WRITEMEMBER(dVdsH);
WRITEMEMBER(massH);
WRITEMEMBER(presH);
WRITEMEMBER(iotaH);
WRITEMEMBER(rmncc);
WRITEMEMBER(rmnss);
WRITEMEMBER(rmnsc);
WRITEMEMBER(rmncs);
WRITEMEMBER(zmnsc);
WRITEMEMBER(zmncs);
WRITEMEMBER(zmncc);
WRITEMEMBER(zmnss);
WRITEMEMBER(lmnsc);
WRITEMEMBER(lmncs);
WRITEMEMBER(lmncc);
WRITEMEMBER(lmnss);
WRITEMEMBER(r_e);
WRITEMEMBER(r_o);
WRITEMEMBER(z_e);
WRITEMEMBER(z_o);
WRITEMEMBER(ru_e);
WRITEMEMBER(ru_o);
WRITEMEMBER(zu_e);
WRITEMEMBER(zu_o);
WRITEMEMBER(rv_e);
WRITEMEMBER(rv_o);
WRITEMEMBER(zv_e);
WRITEMEMBER(zv_o);
WRITEMEMBER(ruFull);
WRITEMEMBER(zuFull);
WRITEMEMBER(r12);
WRITEMEMBER(ru12);
WRITEMEMBER(zu12);
WRITEMEMBER(rs);
WRITEMEMBER(zs);
WRITEMEMBER(gsqrt);
WRITEMEMBER(guu);
WRITEMEMBER(guv);
WRITEMEMBER(gvv);
WRITEMEMBER(bsupu);
WRITEMEMBER(bsupv);
WRITEMEMBER(bsubu);
WRITEMEMBER(bsubv);
WRITEMEMBER(bsubvF);
WRITEMEMBER(total_pressure);
WRITEMEMBER(currv);

return absl::OkStatus();
}

absl::Status vmecpp::VmecInternalResults::LoadInto(
vmecpp::VmecInternalResults& m_obj, H5::H5File& from_file) {
READMEMBER(sign_of_jacobian);
if (from_file.nameExists(absl::StrFormat("%s/%s", H5key, "lamscale"))) {
READMEMBER(lamscale);
} else {
m_obj.lamscale = 1.0;
}
READMEMBER(num_full);
READMEMBER(num_half);
READMEMBER(nZnT_reduced);
READMEMBER(sqrtSH);
READMEMBER(sqrtSF);
READMEMBER(sm);
READMEMBER(sp);
READMEMBER(phipF);
READMEMBER(chipF);
READMEMBER(phipH);

if (H5Lexists(from_file.getId(), absl::StrFormat("%s/chipH", H5key).c_str(),
0) == 1) {
READMEMBER(chipH);
} else {
m_obj.chipH = Eigen::VectorXd::Zero(m_obj.phipH.size());
}

if (H5Lexists(from_file.getId(), absl::StrFormat("%s/currH", H5key).c_str(),
0) == 1) {
READMEMBER(currH);
} else {
m_obj.currH = Eigen::VectorXd::Zero(m_obj.phipH.size());
}
READMEMBER(phiF);
READMEMBER(iotaF);
READMEMBER(spectral_width);
READMEMBER(bvcoH);
READMEMBER(dVdsH);
READMEMBER(massH);
READMEMBER(presH);
READMEMBER(iotaH);
READMEMBER(rmncc);
READMEMBER(rmnss);
READMEMBER(rmnsc);
READMEMBER(rmncs);
READMEMBER(zmnsc);
READMEMBER(zmncs);
READMEMBER(zmncc);
READMEMBER(zmnss);
READMEMBER(lmnsc);
READMEMBER(lmncs);
READMEMBER(lmncc);
READMEMBER(lmnss);
READMEMBER(r_e);
READMEMBER(r_o);
READMEMBER(z_e);
READMEMBER(z_o);
READMEMBER(ru_e);
READMEMBER(ru_o);
READMEMBER(zu_e);
READMEMBER(zu_o);
READMEMBER(rv_e);
READMEMBER(rv_o);
READMEMBER(zv_e);
READMEMBER(zv_o);
READMEMBER(ruFull);
READMEMBER(zuFull);
READMEMBER(r12);
READMEMBER(ru12);
READMEMBER(zu12);
READMEMBER(rs);
READMEMBER(zs);
READMEMBER(gsqrt);
READMEMBER(guu);
READMEMBER(guv);
READMEMBER(gvv);
READMEMBER(bsupu);
READMEMBER(bsupv);
READMEMBER(bsubu);
READMEMBER(bsubv);
READMEMBER(bsubvF);
READMEMBER(total_pressure);
READMEMBER(currv);

return absl::OkStatus();
}

absl::Status vmecpp::RemainingMetric::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(rv12);
WRITEMEMBER(zv12);
WRITEMEMBER(rs12);
WRITEMEMBER(zs12);
WRITEMEMBER(gsu);
WRITEMEMBER(gsv);

return absl::OkStatus();
}

absl::Status vmecpp::RemainingMetric::LoadInto(RemainingMetric& m_obj,
H5::H5File& from_file) {
READMEMBER(rv12);
READMEMBER(zv12);
READMEMBER(rs12);
READMEMBER(zs12);
READMEMBER(gsu);
READMEMBER(gsv);

return absl::OkStatus();
}

absl::Status vmecpp::CylindricalComponentsOfB::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(b_r);
WRITEMEMBER(b_phi);
WRITEMEMBER(b_z);

return absl::OkStatus();
}

absl::Status vmecpp::CylindricalComponentsOfB::LoadInto(
CylindricalComponentsOfB& m_obj, H5::H5File& from_file) {
READMEMBER(b_r);
READMEMBER(b_phi);
READMEMBER(b_z);

return absl::OkStatus();
}

absl::Status vmecpp::BSubSHalf::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(bsubs_half);
return absl::OkStatus();
}

absl::Status vmecpp::BSubSHalf::LoadInto(BSubSHalf& m_obj,
H5::H5File& from_file) {
READMEMBER(bsubs_half);
return absl::OkStatus();
}

absl::Status vmecpp::BSubSFull::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(bsubs_full);
return absl::OkStatus();
}

absl::Status vmecpp::BSubSFull::LoadInto(BSubSFull& m_obj,
H5::H5File& from_file) {
READMEMBER(bsubs_full);
return absl::OkStatus();
}

absl::Status vmecpp::CovariantBDerivatives::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(bsubsu);
WRITEMEMBER(bsubsv);
WRITEMEMBER(bsubuv);
WRITEMEMBER(bsubvu);

return absl::OkStatus();
}

absl::Status vmecpp::CovariantBDerivatives::LoadInto(
CovariantBDerivatives& m_obj, H5::H5File& from_file) {
READMEMBER(bsubsu);
READMEMBER(bsubsv);
READMEMBER(bsubuv);
READMEMBER(bsubvu);

return absl::OkStatus();
}

absl::Status vmecpp::JxBOutFileContents::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(itheta);
WRITEMEMBER(izeta);
WRITEMEMBER(bdotk);
WRITEMEMBER(amaxfor);
WRITEMEMBER(aminfor);
WRITEMEMBER(avforce);
WRITEMEMBER(pprim);
WRITEMEMBER(jdotb);
WRITEMEMBER(bdotb);
WRITEMEMBER(bdotgradv);
WRITEMEMBER(jpar2);
WRITEMEMBER(jperp2);
WRITEMEMBER(phin);
WRITEMEMBER(jsupu3);
WRITEMEMBER(jsupv3);
WRITEMEMBER(jsups3);
WRITEMEMBER(bsupu3);
WRITEMEMBER(bsupv3);
WRITEMEMBER(jcrossb);
WRITEMEMBER(jxb_gradp);
WRITEMEMBER(jdotb_sqrtg);
WRITEMEMBER(sqrtg3);
WRITEMEMBER(bsubu3);
WRITEMEMBER(bsubv3);
WRITEMEMBER(bsubs3);

return absl::OkStatus();
}
absl::Status vmecpp::JxBOutFileContents::LoadInto(JxBOutFileContents& m_obj,
H5::H5File& from_file) {
READMEMBER(itheta);
READMEMBER(izeta);
READMEMBER(bdotk);
READMEMBER(amaxfor);
READMEMBER(aminfor);
READMEMBER(avforce);
READMEMBER(pprim);
READMEMBER(jdotb);
READMEMBER(bdotb);
READMEMBER(bdotgradv);
READMEMBER(jpar2);
READMEMBER(jperp2);
READMEMBER(phin);
READMEMBER(jsupu3);
READMEMBER(jsupv3);
READMEMBER(jsups3);
READMEMBER(bsupu3);
READMEMBER(bsupv3);
READMEMBER(jcrossb);
READMEMBER(jxb_gradp);
READMEMBER(jdotb_sqrtg);
READMEMBER(sqrtg3);
READMEMBER(bsubu3);
READMEMBER(bsubv3);
READMEMBER(bsubs3);
return absl::OkStatus();
}

absl::Status vmecpp::MercierStabilityIntermediateQuantities::WriteTo(
H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(s);
WRITEMEMBER(shear);
WRITEMEMBER(vpp);
WRITEMEMBER(d_pressure_d_s);
WRITEMEMBER(d_toroidal_current_d_s);
WRITEMEMBER(phip_realH);
WRITEMEMBER(phip_realF);
WRITEMEMBER(vp_real);
WRITEMEMBER(torcur);
WRITEMEMBER(gsqrt_full);
WRITEMEMBER(bdotj);
WRITEMEMBER(gpp);
WRITEMEMBER(b2);
WRITEMEMBER(tpp);
WRITEMEMBER(tbb);
WRITEMEMBER(tjb);
WRITEMEMBER(tjj);

return absl::OkStatus();
}
absl::Status vmecpp::MercierStabilityIntermediateQuantities::LoadInto(
MercierStabilityIntermediateQuantities& m_obj, H5::H5File& from_file) {
READMEMBER(s);
READMEMBER(shear);
READMEMBER(vpp);
READMEMBER(d_pressure_d_s);
READMEMBER(d_toroidal_current_d_s);
READMEMBER(phip_realH);
READMEMBER(phip_realF);
READMEMBER(vp_real);
READMEMBER(torcur);
READMEMBER(gsqrt_full);
READMEMBER(bdotj);
READMEMBER(gpp);
READMEMBER(b2);
READMEMBER(tpp);
READMEMBER(tbb);
READMEMBER(tjb);
READMEMBER(tjj);
return absl::OkStatus();
}

absl::Status vmecpp::MercierFileContents::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(s);
WRITEMEMBER(toroidal_flux);
WRITEMEMBER(iota);
WRITEMEMBER(shear);
WRITEMEMBER(d_volume_d_s);
WRITEMEMBER(well);
WRITEMEMBER(toroidal_current);
WRITEMEMBER(d_toroidal_current_d_s);
WRITEMEMBER(pressure);
WRITEMEMBER(d_pressure_d_s);
WRITEMEMBER(DMerc);
WRITEMEMBER(Dshear);
WRITEMEMBER(Dwell);
WRITEMEMBER(Dcurr);
WRITEMEMBER(Dgeod);

return absl::OkStatus();
}
absl::Status vmecpp::MercierFileContents::LoadInto(MercierFileContents& m_obj,
H5::H5File& from_file) {
READMEMBER(s);
READMEMBER(toroidal_flux);
READMEMBER(iota);
READMEMBER(shear);
READMEMBER(d_volume_d_s);
READMEMBER(well);
READMEMBER(toroidal_current);
READMEMBER(d_toroidal_current_d_s);
READMEMBER(pressure);
READMEMBER(d_pressure_d_s);
READMEMBER(DMerc);
READMEMBER(Dshear);
READMEMBER(Dwell);
READMEMBER(Dcurr);
READMEMBER(Dgeod);
return absl::OkStatus();
}

absl::Status vmecpp::Threed1FirstTableIntermediate::WriteTo(
H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(tau);
WRITEMEMBER(beta_vol);
WRITEMEMBER(overr);
WRITEMEMBER(beta_axis);
WRITEMEMBER(presf);
WRITEMEMBER(phipf_loc);
WRITEMEMBER(phi1);
WRITEMEMBER(chi1);
WRITEMEMBER(chi);
WRITEMEMBER(bvcoH);
WRITEMEMBER(bucoH);
WRITEMEMBER(jcurv);
WRITEMEMBER(jcuru);
WRITEMEMBER(presgrad);
WRITEMEMBER(vpphi);
WRITEMEMBER(equif);
WRITEMEMBER(bucof);
WRITEMEMBER(bvcof);

return absl::OkStatus();
}
absl::Status vmecpp::Threed1FirstTableIntermediate::LoadInto(
Threed1FirstTableIntermediate& m_obj, H5::H5File& from_file) {
READMEMBER(tau);
READMEMBER(beta_vol);
READMEMBER(overr);
READMEMBER(beta_axis);
READMEMBER(presf);
READMEMBER(phipf_loc);
READMEMBER(phi1);
READMEMBER(chi1);
READMEMBER(chi);
READMEMBER(bvcoH);
READMEMBER(bucoH);
READMEMBER(jcurv);
READMEMBER(jcuru);
READMEMBER(presgrad);
READMEMBER(vpphi);
READMEMBER(equif);
READMEMBER(bucof);
READMEMBER(bvcof);
return absl::OkStatus();
}

absl::Status vmecpp::Threed1FirstTable::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(s);
WRITEMEMBER(radial_force);
WRITEMEMBER(toroidal_flux);
WRITEMEMBER(iota);
WRITEMEMBER(avg_jsupu);
WRITEMEMBER(avg_jsupv);
WRITEMEMBER(d_volume_d_phi);
WRITEMEMBER(d_pressure_d_phi);
WRITEMEMBER(spectral_width);
WRITEMEMBER(pressure);
WRITEMEMBER(buco_full);
WRITEMEMBER(bvco_full);
WRITEMEMBER(j_dot_b);
WRITEMEMBER(b_dot_b);

return absl::OkStatus();
}
absl::Status vmecpp::Threed1FirstTable::LoadInto(Threed1FirstTable& m_obj,
H5::H5File& from_file) {
READMEMBER(s);
READMEMBER(radial_force);
READMEMBER(toroidal_flux);
READMEMBER(iota);
READMEMBER(avg_jsupu);
READMEMBER(avg_jsupv);
READMEMBER(d_volume_d_phi);
READMEMBER(d_pressure_d_phi);
READMEMBER(spectral_width);
READMEMBER(pressure);
READMEMBER(buco_full);
READMEMBER(bvco_full);
READMEMBER(j_dot_b);
READMEMBER(b_dot_b);
return absl::OkStatus();
}

absl::Status vmecpp::Threed1GeometricAndMagneticQuantitiesIntermediate::WriteTo(
H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(anorm);
WRITEMEMBER(vnorm);
WRITEMEMBER(surf_area);
WRITEMEMBER(circumference_sum);
WRITEMEMBER(rcenin);
WRITEMEMBER(aminr2in);
WRITEMEMBER(bminz2in);
WRITEMEMBER(bminz2);
WRITEMEMBER(sump);
WRITEMEMBER(btor_vac);
WRITEMEMBER(btor1);
WRITEMEMBER(dbtor);
WRITEMEMBER(phat);
WRITEMEMBER(redge);
WRITEMEMBER(delphid_exact);
WRITEMEMBER(musubi);
WRITEMEMBER(rshaf1);
WRITEMEMBER(rshaf2);
WRITEMEMBER(rshaf);
WRITEMEMBER(fpsi0);
WRITEMEMBER(sumbtot);
WRITEMEMBER(sumbtor);
WRITEMEMBER(sumbpol);
WRITEMEMBER(sump20);
WRITEMEMBER(sump2);
WRITEMEMBER(jPS2);
WRITEMEMBER(jpar_perp_sum);
WRITEMEMBER(jparPS_perp_sum);
WRITEMEMBER(s2);
WRITEMEMBER(fac);
WRITEMEMBER(r3v);

return absl::OkStatus();
}
absl::Status
vmecpp::Threed1GeometricAndMagneticQuantitiesIntermediate::LoadInto(
Threed1GeometricAndMagneticQuantitiesIntermediate& m_obj,
H5::H5File& from_file) {
READMEMBER(anorm);
READMEMBER(vnorm);
READMEMBER(surf_area);
READMEMBER(circumference_sum);
READMEMBER(rcenin);
READMEMBER(aminr2in);
READMEMBER(bminz2in);
READMEMBER(bminz2);
READMEMBER(sump);
READMEMBER(btor_vac);
READMEMBER(btor1);
READMEMBER(dbtor);
READMEMBER(phat);
READMEMBER(redge);
READMEMBER(delphid_exact);
READMEMBER(musubi);
READMEMBER(rshaf1);
READMEMBER(rshaf2);
READMEMBER(rshaf);
READMEMBER(fpsi0);
READMEMBER(sumbtot);
READMEMBER(sumbtor);
READMEMBER(sumbpol);
READMEMBER(sump20);
READMEMBER(sump2);
READMEMBER(jPS2);
READMEMBER(jpar_perp_sum);
READMEMBER(jparPS_perp_sum);
READMEMBER(s2);
READMEMBER(fac);
READMEMBER(r3v);
return absl::OkStatus();
}

absl::Status vmecpp::Threed1GeometricAndMagneticQuantities::WriteTo(
H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(toroidal_flux);
WRITEMEMBER(circum_p);
WRITEMEMBER(surf_area_p);
WRITEMEMBER(cross_area_p);
WRITEMEMBER(volume_p);
WRITEMEMBER(Rmajor_p);
WRITEMEMBER(Aminor_p);
WRITEMEMBER(aspect);
WRITEMEMBER(kappa_p);
WRITEMEMBER(rcen);
WRITEMEMBER(aminr1);
WRITEMEMBER(pavg);
WRITEMEMBER(factor);
WRITEMEMBER(b0);
WRITEMEMBER(rmax_surf);
WRITEMEMBER(rmin_surf);
WRITEMEMBER(zmax_surf);
WRITEMEMBER(bmin);
WRITEMEMBER(bmax);
WRITEMEMBER(waist);
WRITEMEMBER(height);
WRITEMEMBER(betapol);
WRITEMEMBER(betatot);
WRITEMEMBER(betator);
WRITEMEMBER(VolAvgB);
WRITEMEMBER(IonLarmor);
WRITEMEMBER(jpar_perp);
WRITEMEMBER(jparPS_perp);
WRITEMEMBER(toroidal_current);
WRITEMEMBER(rbtor);
WRITEMEMBER(rbtor0);
WRITEMEMBER(psi);
WRITEMEMBER(ygeo);
WRITEMEMBER(yinden);
WRITEMEMBER(yellip);
WRITEMEMBER(ytrian);
WRITEMEMBER(yshift);
WRITEMEMBER(loc_jpar_perp);
WRITEMEMBER(loc_jparPS_perp);

return absl::OkStatus();
}
absl::Status vmecpp::Threed1GeometricAndMagneticQuantities::LoadInto(
Threed1GeometricAndMagneticQuantities& m_obj, H5::H5File& from_file) {
READMEMBER(toroidal_flux);
READMEMBER(circum_p);
READMEMBER(surf_area_p);
READMEMBER(cross_area_p);
READMEMBER(volume_p);
READMEMBER(Rmajor_p);
READMEMBER(Aminor_p);
READMEMBER(aspect);
READMEMBER(kappa_p);
READMEMBER(rcen);
READMEMBER(aminr1);
READMEMBER(pavg);
READMEMBER(factor);
READMEMBER(b0);
READMEMBER(rmax_surf);
READMEMBER(rmin_surf);
READMEMBER(zmax_surf);
READMEMBER(bmin);
READMEMBER(bmax);
READMEMBER(waist);
READMEMBER(height);
READMEMBER(betapol);
READMEMBER(betatot);
READMEMBER(betator);
READMEMBER(VolAvgB);
READMEMBER(IonLarmor);
READMEMBER(jpar_perp);
READMEMBER(jparPS_perp);
READMEMBER(toroidal_current);
READMEMBER(rbtor);
READMEMBER(rbtor0);
READMEMBER(psi);
READMEMBER(ygeo);
READMEMBER(yinden);
READMEMBER(yellip);
READMEMBER(ytrian);
READMEMBER(yshift);
READMEMBER(loc_jpar_perp);
READMEMBER(loc_jparPS_perp);

return absl::OkStatus();
}

absl::Status vmecpp::Threed1Volumetrics::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(int_p);
WRITEMEMBER(avg_p);
WRITEMEMBER(int_bpol);
WRITEMEMBER(avg_bpol);
WRITEMEMBER(int_btor);
WRITEMEMBER(avg_btor);
WRITEMEMBER(int_modb);
WRITEMEMBER(avg_modb);
WRITEMEMBER(int_ekin);
WRITEMEMBER(avg_ekin);

return absl::OkStatus();
}

absl::Status vmecpp::Threed1Volumetrics::LoadInto(Threed1Volumetrics& m_obj,
H5::H5File& from_file) {
READMEMBER(int_p);
READMEMBER(avg_p);
READMEMBER(int_bpol);
READMEMBER(avg_bpol);
READMEMBER(int_btor);
READMEMBER(avg_btor);
READMEMBER(int_modb);
READMEMBER(avg_modb);
READMEMBER(int_ekin);
READMEMBER(avg_ekin);
return absl::OkStatus();
}

absl::Status vmecpp::Threed1AxisGeometry::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(raxis_symm);
WRITEMEMBER(zaxis_symm);
WRITEMEMBER(raxis_asym);
WRITEMEMBER(zaxis_asym);

return absl::OkStatus();
}

absl::Status vmecpp::Threed1AxisGeometry::LoadInto(Threed1AxisGeometry& m_obj,
H5::H5File& from_file) {
READMEMBER(raxis_symm);
READMEMBER(zaxis_symm);
READMEMBER(raxis_asym);
READMEMBER(zaxis_asym);
return absl::OkStatus();
}

absl::Status vmecpp::Threed1FreeBoundary::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(rb);
WRITEMEMBER(phib);
WRITEMEMBER(zb);
WRITEMEMBER(bsqmhdi);
WRITEMEMBER(bsqvaci);
WRITEMEMBER(bsqmhdf);
WRITEMEMBER(bsqvacf);
WRITEMEMBER(bredge);
WRITEMEMBER(bpedge);
WRITEMEMBER(bzedge);
WRITEMEMBER(brv);
WRITEMEMBER(bphiv);
WRITEMEMBER(bzv);

return absl::OkStatus();
}

absl::Status vmecpp::Threed1FreeBoundary::LoadInto(Threed1FreeBoundary& m_obj,
H5::H5File& from_file) {

if (H5Lexists(from_file.getId(), H5key, 0) != 1) {
return absl::OkStatus();
}
READMEMBER(rb);
READMEMBER(phib);
READMEMBER(zb);
READMEMBER(bsqmhdi);
READMEMBER(bsqvaci);
READMEMBER(bsqmhdf);
READMEMBER(bsqvacf);
READMEMBER(bredge);
READMEMBER(bpedge);
READMEMBER(bzedge);
READMEMBER(brv);
READMEMBER(bphiv);
READMEMBER(bzv);
return absl::OkStatus();
}

absl::Status vmecpp::Threed1Betas::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(betatot);
WRITEMEMBER(betapol);
WRITEMEMBER(betator);
WRITEMEMBER(rbtor);
WRITEMEMBER(betaxis);
WRITEMEMBER(betstr);

return absl::OkStatus();
}
absl::Status vmecpp::Threed1Betas::LoadInto(Threed1Betas& m_obj,
H5::H5File& from_file) {
READMEMBER(betatot);
READMEMBER(betapol);
READMEMBER(betator);
READMEMBER(rbtor);
READMEMBER(betaxis);
READMEMBER(betstr);
return absl::OkStatus();
}

absl::Status vmecpp::Threed1ShafranovIntegrals::WriteTo(
H5::H5File& file) const {
file.createGroup(this->H5key);
WRITEMEMBER(scaling_ratio);
WRITEMEMBER(r_lao);
WRITEMEMBER(f_lao);
WRITEMEMBER(f_geo);
WRITEMEMBER(smaleli);
WRITEMEMBER(betai);
WRITEMEMBER(musubi);
WRITEMEMBER(lambda);
WRITEMEMBER(s11);
WRITEMEMBER(s12);
WRITEMEMBER(s13);
WRITEMEMBER(s2);
WRITEMEMBER(s3);
WRITEMEMBER(delta1);
WRITEMEMBER(delta2);
WRITEMEMBER(delta3);

return absl::OkStatus();
}
absl::Status vmecpp::Threed1ShafranovIntegrals::LoadInto(
Threed1ShafranovIntegrals& m_obj, H5::H5File& from_file) {
READMEMBER(scaling_ratio);
READMEMBER(r_lao);
READMEMBER(f_lao);
READMEMBER(f_geo);
READMEMBER(smaleli);
READMEMBER(betai);
READMEMBER(musubi);
READMEMBER(lambda);
READMEMBER(s11);
READMEMBER(s12);
READMEMBER(s13);
READMEMBER(s2);
READMEMBER(s3);
READMEMBER(delta1);
READMEMBER(delta2);
READMEMBER(delta3);
return absl::OkStatus();
}

absl::Status vmecpp::WOutFileContents::WriteTo(H5::H5File& file) const {
file.createGroup(this->H5key);

WRITEMEMBER(version_);

WRITEMEMBER(signgs);
WRITEMEMBER(gamma);
WRITEMEMBER(pcurr_type);
WRITEMEMBER(pmass_type);
WRITEMEMBER(piota_type);
WRITEMEMBER(am);
WRITEMEMBER(ac);
WRITEMEMBER(ai);
WRITEMEMBER(am_aux_s);
WRITEMEMBER(am_aux_f);
WRITEMEMBER(ac_aux_s);
WRITEMEMBER(ac_aux_f);
WRITEMEMBER(ai_aux_s);
WRITEMEMBER(ai_aux_f);
WRITEMEMBER(nfp);
WRITEMEMBER(mpol);
WRITEMEMBER(ntor);
WRITEMEMBER(lasym);
WRITEMEMBER(ns);
WRITEMEMBER(ftolv);
WRITEMEMBER(niter);
WRITEMEMBER(lfreeb);
WRITEMEMBER(mgrid_file);
WRITEMEMBER(nextcur);
WRITEMEMBER(extcur);
WRITEMEMBER(mgrid_mode);
WRITEMEMBER(wb);
WRITEMEMBER(wp);
WRITEMEMBER(rmax_surf);
WRITEMEMBER(rmin_surf);
WRITEMEMBER(zmax_surf);
WRITEMEMBER(mnmax);
WRITEMEMBER(mnmax_nyq);
WRITEMEMBER(ier_flag);
WRITEMEMBER(aspect);
WRITEMEMBER(betatotal);
WRITEMEMBER(betapol);
WRITEMEMBER(betator);
WRITEMEMBER(betaxis);
WRITEMEMBER(b0);
WRITEMEMBER(rbtor0);
WRITEMEMBER(rbtor);
WRITEMEMBER(IonLarmor);
WRITEMEMBER(volavgB);
WRITEMEMBER(ctor);
WRITEMEMBER(Aminor_p);
WRITEMEMBER(Rmajor_p);
WRITEMEMBER(volume);
WRITEMEMBER(fsqr);
WRITEMEMBER(fsqz);
WRITEMEMBER(fsql);
WRITEMEMBER(iotaf);
WRITEMEMBER(q_factor);
WRITEMEMBER(presf);
WRITEMEMBER(phi);
WRITEMEMBER(phipf);
WRITEMEMBER(chi);
WRITEMEMBER(chipf);
WRITEMEMBER(jcuru);
WRITEMEMBER(jcurv);
WRITEMEMBER(iotas);
WRITEMEMBER(mass);
WRITEMEMBER(pres);
WRITEMEMBER(beta_vol);
WRITEMEMBER(buco);
WRITEMEMBER(bvco);
WRITEMEMBER(vp);
WRITEMEMBER(specw);
WRITEMEMBER(phips);
WRITEMEMBER(over_r);
WRITEMEMBER(jdotb);
WRITEMEMBER(bdotb);
WRITEMEMBER(bdotgradv);
WRITEMEMBER(DMerc);
WRITEMEMBER(DShear);
WRITEMEMBER(DWell);
WRITEMEMBER(DCurr);
WRITEMEMBER(DGeod);
WRITEMEMBER(equif);
WRITEMEMBER(curlabel);
WRITEMEMBER(potvac);
WRITEMEMBER(xmpot);
WRITEMEMBER(xnpot);
WRITEMEMBER(xm);
WRITEMEMBER(xn);
WRITEMEMBER(xm_nyq);
WRITEMEMBER(xn_nyq);
WRITEMEMBER(raxis_cc);
WRITEMEMBER(zaxis_cs);
WRITEMEMBER(rmnc);
WRITEMEMBER(zmns);
WRITEMEMBER(lmns_full);
WRITEMEMBER(lmns);
WRITEMEMBER(gmnc);
WRITEMEMBER(bmnc);
WRITEMEMBER(bsubumnc);
WRITEMEMBER(bsubvmnc);
WRITEMEMBER(bsubsmns);
WRITEMEMBER(bsubsmns_full);
WRITEMEMBER(bsupumnc);
WRITEMEMBER(bsupvmnc);
WRITEMEMBER(currumnc);
WRITEMEMBER(currvmnc);
WRITEMEMBER(raxis_cs);
WRITEMEMBER(zaxis_cc);
WRITEMEMBER(rmns);
WRITEMEMBER(zmnc);
WRITEMEMBER(lmnc_full);
WRITEMEMBER(lmnc);
WRITEMEMBER(gmns);
WRITEMEMBER(bmns);
WRITEMEMBER(bsubumns);
WRITEMEMBER(bsubvmns);
WRITEMEMBER(bsubsmnc);
WRITEMEMBER(bsubsmnc_full);
WRITEMEMBER(bsupumns);
WRITEMEMBER(bsupvmns);
WRITEMEMBER(currumns);
WRITEMEMBER(currvmns);

return absl::OkStatus();
}

absl::Status vmecpp::WOutFileContents::LoadInto(WOutFileContents& m_obj,
H5::H5File& from_file) {

if (from_file.nameExists(absl::StrFormat("%s/%s", H5key, "version_"))) {
READMEMBER(version_);
} else {
std::string version_str;
ReadH5Dataset(version_str, absl::StrFormat("%s/%s", H5key, "version"),
from_file);
m_obj.version_ = std::stod(version_str);
}

READMEMBER_COMPAT(signgs, "sign_of_jacobian");
READMEMBER(gamma);
READMEMBER(pcurr_type);
READMEMBER(pmass_type);
READMEMBER(piota_type);
READMEMBER(am);
READMEMBER(ac);
READMEMBER(ai);
READMEMBER(am_aux_s);
READMEMBER(am_aux_f);
READMEMBER(ac_aux_s);
READMEMBER(ac_aux_f);
READMEMBER(ai_aux_s);
READMEMBER(ai_aux_f);
READMEMBER(nfp);
READMEMBER(mpol);
READMEMBER(ntor);
READMEMBER(lasym);
READMEMBER(ns);
READMEMBER(ftolv);
READMEMBER_COMPAT(niter, "maximum_iterations");
READMEMBER(lfreeb);
READMEMBER(mgrid_file);

if (m_obj.lfreeb) {
READMEMBER(nextcur);
READMEMBER(extcur);
} else {
m_obj.nextcur = 0;
m_obj.extcur = Eigen::Vector<double, 0>::Zero();
}
READMEMBER(mgrid_mode);
READMEMBER(wb);
READMEMBER(wp);
READMEMBER(rmax_surf);
READMEMBER(rmin_surf);
READMEMBER(zmax_surf);
READMEMBER(mnmax);
READMEMBER(mnmax_nyq);
READMEMBER(ier_flag);
READMEMBER(aspect);
READMEMBER_COMPAT(betatotal, "betatot");
READMEMBER(betapol);
READMEMBER(betator);
READMEMBER(betaxis);
READMEMBER(b0);
READMEMBER(rbtor0);
READMEMBER(rbtor);
READMEMBER(IonLarmor);
READMEMBER_COMPAT(volavgB, "VolAvgB");
READMEMBER(ctor);
READMEMBER(Aminor_p);
READMEMBER(Rmajor_p);
READMEMBER_COMPAT(volume, "volume_p");
READMEMBER(fsqr);
READMEMBER(fsqz);
READMEMBER(fsql);
READMEMBER_COMPAT(iotaf, "iota_full");
READMEMBER_COMPAT(q_factor, "safety_factor");
READMEMBER_COMPAT(presf, "pressure_full");
READMEMBER_COMPAT(phi, "toroidal_flux");
READMEMBER(phipf);
READMEMBER_COMPAT(chi, "poloidal_flux");
READMEMBER(chipf);
READMEMBER(jcuru);
READMEMBER(jcurv);

READMEMBER_COMPAT(specw, "spectral_width");

if (from_file.nameExists(absl::StrFormat("%s/%s", H5key, "iotas"))) {
READMEMBER(iotas);
READMEMBER(mass);
READMEMBER(pres);
READMEMBER(beta_vol);
READMEMBER(buco);
READMEMBER(bvco);
READMEMBER(vp);
READMEMBER(phips);
READMEMBER(over_r);
} else {

auto ReadHalfGridCompat = [&](Eigen::VectorXd& dest,
const std::string& old_name) {
Eigen::VectorXd tmp;
ReadH5Dataset(tmp, absl::StrFormat("%s/%s", H5key, old_name), from_file);
dest = Eigen::VectorXd::Zero(tmp.size() + 1);
dest.tail(tmp.size()) = tmp;
};
ReadHalfGridCompat(m_obj.iotas, "iota_half");
ReadHalfGridCompat(m_obj.mass, "mass");
ReadHalfGridCompat(m_obj.pres, "pressure_half");
ReadHalfGridCompat(m_obj.beta_vol, "beta");
ReadHalfGridCompat(m_obj.buco, "buco");
ReadHalfGridCompat(m_obj.bvco, "bvco");
ReadHalfGridCompat(m_obj.vp, "dVds");
ReadHalfGridCompat(m_obj.phips, "phips");
ReadHalfGridCompat(m_obj.over_r, "overr");
}
READMEMBER(jdotb);

READMEMBER_OPTIONAL(bdotb);
READMEMBER(bdotgradv);
READMEMBER(DMerc);
READMEMBER_COMPAT(DShear, "Dshear");
READMEMBER_COMPAT(DWell, "Dwell");
READMEMBER_COMPAT(DCurr, "Dcurr");
READMEMBER_COMPAT(DGeod, "Dgeod");
READMEMBER(equif);
READMEMBER(curlabel);
READMEMBER(potvac);
READMEMBER_OPTIONAL(xmpot);
READMEMBER_OPTIONAL(xnpot);
READMEMBER(xm);
READMEMBER(xn);
READMEMBER(xm_nyq);
READMEMBER(xn_nyq);
READMEMBER_COMPAT(raxis_cc, "raxis_c");
READMEMBER_COMPAT(zaxis_cs, "zaxis_s");

auto ReadAndTranspose2D = [&](RowMatrixXd& dest, const std::string& name) {
RowMatrixXd tmp;
ReadH5Dataset(tmp, absl::StrFormat("%s/%s", H5key, name), from_file);

if (tmp.rows() == m_obj.ns &&
(tmp.cols() == m_obj.mnmax || tmp.cols() == m_obj.mnmax_nyq)) {
dest = tmp.transpose();
} else {
dest = tmp;
}
};
auto ReadAndTransposePadHalfGrid2D =
[&](RowMatrixXd& dest, const std::string& name, int mnsize) {
RowMatrixXd tmp;
ReadH5Dataset(tmp, absl::StrFormat("%s/%s", H5key, name), from_file);

if (tmp.rows() == m_obj.ns - 1 && tmp.cols() == mnsize) {
dest = RowMatrixXd::Zero(mnsize, m_obj.ns);
dest.rightCols(m_obj.ns - 1) = tmp.transpose();
} else if (tmp.rows() == m_obj.ns && tmp.cols() == mnsize) {

dest = tmp.transpose();
} else {
dest = tmp;
}
};

ReadAndTranspose2D(m_obj.rmnc, "rmnc");
ReadAndTranspose2D(m_obj.zmns, "zmns");
ReadAndTranspose2D(m_obj.lmns_full, "lmns_full");
ReadAndTransposePadHalfGrid2D(m_obj.lmns, "lmns", m_obj.mnmax);
ReadAndTransposePadHalfGrid2D(m_obj.gmnc, "gmnc", m_obj.mnmax_nyq);
ReadAndTransposePadHalfGrid2D(m_obj.bmnc, "bmnc", m_obj.mnmax_nyq);
ReadAndTransposePadHalfGrid2D(m_obj.bsubumnc, "bsubumnc", m_obj.mnmax_nyq);
ReadAndTransposePadHalfGrid2D(m_obj.bsubvmnc, "bsubvmnc", m_obj.mnmax_nyq);
ReadAndTransposePadHalfGrid2D(m_obj.bsubsmns, "bsubsmns", m_obj.mnmax_nyq);
READMEMBER(bsubsmns_full);
ReadAndTransposePadHalfGrid2D(m_obj.bsupumnc, "bsupumnc", m_obj.mnmax_nyq);
ReadAndTransposePadHalfGrid2D(m_obj.bsupvmnc, "bsupvmnc", m_obj.mnmax_nyq);
if (from_file.nameExists(absl::StrFormat("%s/currumnc", H5key))) {
ReadAndTranspose2D(m_obj.currumnc, "currumnc");
}
if (from_file.nameExists(absl::StrFormat("%s/currvmnc", H5key))) {
ReadAndTranspose2D(m_obj.currvmnc, "currvmnc");
}
READMEMBER_COMPAT(raxis_cs, "raxis_s");
READMEMBER_COMPAT(zaxis_cc, "zaxis_c");
ReadAndTranspose2D(m_obj.rmns, "rmns");
ReadAndTranspose2D(m_obj.zmnc, "zmnc");
ReadAndTranspose2D(m_obj.lmnc_full, "lmnc_full");
ReadAndTransposePadHalfGrid2D(m_obj.lmnc, "lmnc", m_obj.mnmax);
ReadAndTransposePadHalfGrid2D(m_obj.gmns, "gmns", m_obj.mnmax_nyq);
ReadAndTransposePadHalfGrid2D(m_obj.bmns, "bmns", m_obj.mnmax_nyq);
ReadAndTransposePadHalfGrid2D(m_obj.bsubumns, "bsubumns", m_obj.mnmax_nyq);
ReadAndTransposePadHalfGrid2D(m_obj.bsubvmns, "bsubvmns", m_obj.mnmax_nyq);
ReadAndTransposePadHalfGrid2D(m_obj.bsubsmnc, "bsubsmnc", m_obj.mnmax_nyq);
READMEMBER(bsubsmnc_full);
ReadAndTransposePadHalfGrid2D(m_obj.bsupumns, "bsupumns", m_obj.mnmax_nyq);
ReadAndTransposePadHalfGrid2D(m_obj.bsupvmns, "bsupvmns", m_obj.mnmax_nyq);
if (from_file.nameExists(absl::StrFormat("%s/currumns", H5key))) {
ReadAndTranspose2D(m_obj.currumns, "currumns");
}
if (from_file.nameExists(absl::StrFormat("%s/currvmns", H5key))) {
ReadAndTranspose2D(m_obj.currvmns, "currvmns");
}

return absl::OkStatus();
}

#undef WRITEMEMBER
#undef READMEMBER

absl::Status vmecpp::OutputQuantities::WriteTo(H5::H5File& file) const {
absl::Status status;

status = vmec_internal_results.WriteTo(file);
if (!status.ok()) {
return status;
}

status = remaining_metric.WriteTo(file);
if (!status.ok()) {
return status;
}

status = b_cylindrical.WriteTo(file);
if (!status.ok()) {
return status;
}

status = bsubs_half.WriteTo(file);
if (!status.ok()) {
return status;
}

status = bsubs_full.WriteTo(file);
if (!status.ok()) {
return status;
}

status = covariant_b_derivatives.WriteTo(file);
if (!status.ok()) {
return status;
}

status = jxbout.WriteTo(file);
if (!status.ok()) {
return status;
}

status = mercier_intermediate.WriteTo(file);
if (!status.ok()) {
return status;
}

status = mercier.WriteTo(file);
if (!status.ok()) {
return status;
}

status = threed1_first_table_intermediate.WriteTo(file);
if (!status.ok()) {
return status;
}

status = threed1_first_table.WriteTo(file);
if (!status.ok()) {
return status;
}

status = threed1_geometric_magnetic_intermediate.WriteTo(file);
if (!status.ok()) {
return status;
}

status = threed1_geometric_magnetic.WriteTo(file);
if (!status.ok()) {
return status;
}

status = threed1_volumetrics.WriteTo(file);
if (!status.ok()) {
return status;
}

status = threed1_axis.WriteTo(file);
if (!status.ok()) {
return status;
}

status = threed1_free_boundary.WriteTo(file);
if (!status.ok()) {
return status;
}

status = threed1_betas.WriteTo(file);
if (!status.ok()) {
return status;
}

status = threed1_shafranov_integrals.WriteTo(file);
if (!status.ok()) {
return status;
}

status = wout.WriteTo(file);
if (!status.ok()) {
return status;
}

status = indata.WriteTo(file);
if (!status.ok()) {
return status;
}

return absl::OkStatus();
}

absl::StatusOr<vmecpp::OutputQuantities> vmecpp::OutputQuantities::ReadFrom(
H5::H5File& file) {
OutputQuantities oq;
absl::Status status;

status = decltype(oq.vmec_internal_results)::LoadInto(
oq.vmec_internal_results, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.remaining_metric)::LoadInto(oq.remaining_metric, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.b_cylindrical)::LoadInto(oq.b_cylindrical, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.bsubs_half)::LoadInto(oq.bsubs_half, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.bsubs_full)::LoadInto(oq.bsubs_full, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.covariant_b_derivatives)::LoadInto(
oq.covariant_b_derivatives, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.jxbout)::LoadInto(oq.jxbout, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.mercier_intermediate)::LoadInto(oq.mercier_intermediate,
file);
if (!status.ok()) {
return status;
}

status = decltype(oq.mercier)::LoadInto(oq.mercier, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.threed1_first_table_intermediate)::LoadInto(
oq.threed1_first_table_intermediate, file);
if (!status.ok()) {
return status;
}

status =
decltype(oq.threed1_first_table)::LoadInto(oq.threed1_first_table, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.threed1_geometric_magnetic_intermediate)::LoadInto(
oq.threed1_geometric_magnetic_intermediate, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.threed1_geometric_magnetic)::LoadInto(
oq.threed1_geometric_magnetic, file);
if (!status.ok()) {
return status;
}

status =
decltype(oq.threed1_volumetrics)::LoadInto(oq.threed1_volumetrics, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.threed1_axis)::LoadInto(oq.threed1_axis, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.threed1_free_boundary)::LoadInto(
oq.threed1_free_boundary, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.threed1_betas)::LoadInto(oq.threed1_betas, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.threed1_shafranov_integrals)::LoadInto(
oq.threed1_shafranov_integrals, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.wout)::LoadInto(oq.wout, file);
if (!status.ok()) {
return status;
}

status = decltype(oq.indata)::LoadInto(oq.indata, file);
if (!status.ok()) {
return status;
}

return oq;
}

absl::Status vmecpp::OutputQuantities::Save(
const std::filesystem::path& path) const {
try {
H5::H5File file(path, H5F_ACC_TRUNC);
return WriteTo(file);
} catch (const H5::Exception& exception) {
return absl::InternalError(
absl::StrFormat("could not write '%s': %s: %s", path.string(),
exception.getFuncName(), exception.getDetailMsg()));
}
}

absl::StatusOr<vmecpp::OutputQuantities> vmecpp::OutputQuantities::Load(
const std::filesystem::path& path) {
try {
H5::H5File file(path, H5F_ACC_RDONLY);
return ReadFrom(file);
} catch (const H5::Exception& exception) {
return absl::InternalError(
absl::StrFormat("could not read '%s': %s: %s", path.string(),
exception.getFuncName(), exception.getDetailMsg()));
}
}

vmecpp::Threed1FreeBoundary vmecpp::ComputeThreed1FreeBoundary(
const Sizes& s, const FlowControl& fc,
const HandoverStorage& handover_storage,
const VmecInternalResults& vmec_internal_results,
const CylindricalComponentsOfB& b_cylindrical) {
Threed1FreeBoundary result;

const int num_zeta = s.nZeta;

const int num_theta = s.nThetaEff;
result.rb = RowMatrixXd::Zero(num_zeta, num_theta);
result.phib = RowMatrixXd::Zero(num_zeta, num_theta);
result.zb = RowMatrixXd::Zero(num_zeta, num_theta);
result.bsqmhdi = RowMatrixXd::Zero(num_zeta, num_theta);
result.bsqvaci = RowMatrixXd::Zero(num_zeta, num_theta);
result.bsqmhdf = RowMatrixXd::Zero(num_zeta, num_theta);
result.bsqvacf = RowMatrixXd::Zero(num_zeta, num_theta);
result.bredge = RowMatrixXd::Zero(num_zeta, num_theta);
result.bpedge = RowMatrixXd::Zero(num_zeta, num_theta);
result.bzedge = RowMatrixXd::Zero(num_zeta, num_theta);
result.brv = RowMatrixXd::Zero(num_zeta, num_theta);
result.bphiv = RowMatrixXd::Zero(num_zeta, num_theta);
result.bzv = RowMatrixXd::Zero(num_zeta, num_theta);

if (handover_storage.vacuum_magnetic_pressure.size() != s.nZnT) {
return result;
}

const int last_full = fc.ns - 1;
const int last_half = fc.ns - 2;
const int previous_half = fc.ns - 3;

for (int k = 0; k < num_zeta; ++k) {
const double zeta = 2.0 * M_PI * k / (num_zeta * s.nfp);
for (int l = 0; l < num_theta; ++l) {

const int kl = k * s.nThetaEff + l;
const int lk = l * num_zeta + k;

const int boundary = last_full * s.nZnT + kl;
result.rb(k, l) = vmec_internal_results.r_e(boundary) +
vmec_internal_results.r_o(boundary);
result.zb(k, l) = vmec_internal_results.z_e(boundary) +
vmec_internal_results.z_o(boundary);
result.phib(k, l) = zeta;

result.bsqmhdi(k, l) =
handover_storage.initial_plasma_pressure_at_boundary[kl];
result.bsqvaci(k, l) =
handover_storage.initial_vacuum_pressure_at_boundary[lk];
result.bsqmhdf(k, l) = handover_storage.edge_total_pressure[kl];
result.bsqvacf(k, l) = handover_storage.vacuum_magnetic_pressure[lk];

result.bredge(k, l) = 1.5 * b_cylindrical.b_r(last_half, kl) -
0.5 * b_cylindrical.b_r(previous_half, kl);
result.bpedge(k, l) = 1.5 * b_cylindrical.b_phi(last_half, kl) -
0.5 * b_cylindrical.b_phi(previous_half, kl);
result.bzedge(k, l) = 1.5 * b_cylindrical.b_z(last_half, kl) -
0.5 * b_cylindrical.b_z(previous_half, kl);

result.brv(k, l) = handover_storage.vacuum_b_r[lk];
result.bphiv(k, l) = handover_storage.vacuum_b_phi[lk];
result.bzv(k, l) = handover_storage.vacuum_b_z[lk];
}
}

return result;
}

vmecpp::OutputQuantities vmecpp::ComputeOutputQuantities(
const int sign_of_jacobian, const VmecINDATA& indata, const Sizes& s,
const FlowControl& fc, const VmecConstants& constants,
const FourierBasisFastPoloidal& t, const HandoverStorage& h,
const std::string& mgrid_mode,
const std::vector<std::string>& coil_group_names,
const std::vector<std::unique_ptr<RadialPartitioning>>& radial_partitioning,
const std::vector<std::unique_ptr<FourierGeometry>>& decomposed_x,
const std::vector<std::unique_ptr<IdealMhdModel>>& models_from_threads,
const std::vector<std::unique_ptr<RadialProfiles>>& radial_profiles,
const VmecCheckpoint& checkpoint, VacuumPressureState vacuum_pressure_state,
VmecStatus vmec_status, int iter2) {
OutputQuantities output_quantities;

output_quantities.vmec_internal_results = GatherDataFromThreads(
sign_of_jacobian, s, fc, constants, radial_partitioning, decomposed_x,
models_from_threads, radial_profiles);

if (vmec_status == VmecStatus::NORMAL_TERMINATION ||
vmec_status == VmecStatus::SUCCESSFUL_TERMINATION ||
indata.return_outputs_even_if_not_converged) {
MeshBledingBSubZeta(
s, fc,
output_quantities.vmec_internal_results);

const PoloidalCurrentToFixBSubV pctf = ComputePoloidalCurrentToFixBSubV(
s, output_quantities.vmec_internal_results);
FixupPoloidalCurrent(
s, pctf,
output_quantities.vmec_internal_results);

if (checkpoint == VmecCheckpoint::BCOVAR_FILEOUT) {
return output_quantities;
}

RecomputeToroidalFlux(
fc,
output_quantities.vmec_internal_results);

output_quantities.remaining_metric =
ComputeRemainingMetric(s, output_quantities.vmec_internal_results);
output_quantities.b_cylindrical =
BCylindricalComponents(s, output_quantities.vmec_internal_results,
output_quantities.remaining_metric);
output_quantities.bsubs_half =
ComputeBSubSOnHalfGrid(s, output_quantities.vmec_internal_results,
output_quantities.remaining_metric);

if (checkpoint == VmecCheckpoint::BSS) {
return output_quantities;
}

output_quantities.bsubs_full =
PutBSubSOnFullGrid(s, output_quantities.vmec_internal_results,
output_quantities.bsubs_half);

SymmetryDecomposedCovariantB decomposed_bcov =
DecomposeCovariantBBySymmetry(s,
output_quantities.vmec_internal_results,
output_quantities.bsubs_full);

output_quantities.covariant_b_derivatives = LowPassFilterCovariantB(
s, t, decomposed_bcov,
output_quantities.vmec_internal_results);

if (checkpoint == VmecCheckpoint::LOWPASS_BCOVARIANT) {
return output_quantities;
}

ExtrapolateBSubS(s, fc,  output_quantities.bsubs_full);

if (checkpoint == VmecCheckpoint::EXTRAPOLATE_BSUBS) {
return output_quantities;
}

output_quantities.jxbout = ComputeJxBOutputFileContents(
s, fc, output_quantities.vmec_internal_results,
output_quantities.bsubs_full, output_quantities.covariant_b_derivatives,
indata.return_outputs_even_if_not_converged, vmec_status);

if (checkpoint == VmecCheckpoint::JXBOUT) {
return output_quantities;
}

output_quantities.mercier_intermediate =
ComputeIntermediateMercierQuantities(
s, fc, output_quantities.vmec_internal_results,
output_quantities.jxbout);

output_quantities.mercier =
ComputeMercierStability(fc, output_quantities.vmec_internal_results,
output_quantities.mercier_intermediate);

if (checkpoint == VmecCheckpoint::MERCIER) {
return output_quantities;
}

output_quantities.threed1_first_table_intermediate =
ComputeIntermediateThreed1FirstTableQuantities(
s, fc, output_quantities.vmec_internal_results);

output_quantities.threed1_first_table = ComputeThreed1FirstTable(
fc, output_quantities.vmec_internal_results, output_quantities.jxbout,
output_quantities.threed1_first_table_intermediate);

if (checkpoint == VmecCheckpoint::THREED1_FIRST_TABLE) {
return output_quantities;
}

output_quantities.threed1_geometric_magnetic_intermediate =
ComputeIntermediateThreed1GeometricMagneticQuantities(
s, fc, h, output_quantities.vmec_internal_results,
output_quantities.jxbout,
output_quantities.threed1_first_table_intermediate,
vacuum_pressure_state);

output_quantities.threed1_geometric_magnetic =
ComputeThreed1GeometricMagneticQuantities(
s, fc, h, output_quantities.vmec_internal_results,
output_quantities.jxbout,
output_quantities.threed1_first_table_intermediate,
output_quantities.threed1_geometric_magnetic_intermediate);

if (checkpoint == VmecCheckpoint::THREED1_GEOMAG) {
return output_quantities;
}

output_quantities.threed1_volumetrics = ComputeThreed1Volumetrics(
output_quantities.threed1_geometric_magnetic_intermediate,
output_quantities.threed1_geometric_magnetic);

if (checkpoint == VmecCheckpoint::THREED1_VOLUMETRICS) {
return output_quantities;
}

output_quantities.threed1_axis = ComputeThreed1AxisGeometry(
s, t, output_quantities.vmec_internal_results);

if (checkpoint == VmecCheckpoint::THREED1_AXIS) {
return output_quantities;
}

output_quantities.threed1_betas = ComputeThreed1Betas(
h, output_quantities.threed1_first_table_intermediate,
output_quantities.threed1_geometric_magnetic_intermediate,
output_quantities.threed1_geometric_magnetic);

if (checkpoint == VmecCheckpoint::THREED1_BETAS) {
return output_quantities;
}

output_quantities.threed1_shafranov_integrals =
ComputeThreed1ShafranovIntegrals(
s, fc, h, output_quantities.vmec_internal_results,
output_quantities.threed1_geometric_magnetic_intermediate,
output_quantities.threed1_geometric_magnetic,
vacuum_pressure_state);

if (checkpoint == VmecCheckpoint::THREED1_SHAFRANOV_INTEGRALS) {
return output_quantities;
}

output_quantities.wout = ComputeWOutFileContents(
indata, s, t, fc, constants, h, mgrid_mode, coil_group_names,
output_quantities.vmec_internal_results,
output_quantities.bsubs_half, output_quantities.bsubs_full,
output_quantities.mercier, output_quantities.jxbout,
output_quantities.threed1_first_table_intermediate,
output_quantities.threed1_first_table,
output_quantities.threed1_geometric_magnetic,
output_quantities.threed1_axis, output_quantities.threed1_betas,
output_quantities.threed1_free_boundary, vmec_status, iter2);
}

output_quantities.threed1_free_boundary = ComputeThreed1FreeBoundary(
s, fc, h, output_quantities.vmec_internal_results,
output_quantities.b_cylindrical);

output_quantities.indata = indata;

return output_quantities;
}

vmecpp::VmecInternalResults vmecpp::GatherDataFromThreads(
const int sign_of_jacobian, const Sizes& s, const FlowControl& fc,
const VmecConstants& constants,
const std::vector<std::unique_ptr<RadialPartitioning>>& radial_partitioning,
const std::vector<std::unique_ptr<FourierGeometry>>& decomposed_x,
const std::vector<std::unique_ptr<IdealMhdModel>>& models_from_threads,
const std::vector<std::unique_ptr<RadialProfiles>>& radial_profiles) {
VmecInternalResults results;

results.sign_of_jacobian = sign_of_jacobian;
results.lamscale = constants.lamscale;

results.num_half = fc.ns - 1;
results.num_full = fc.ns;

results.nZnT_reduced = s.nThetaReduced * s.nZeta;

results.sqrtSH = VectorXd::Zero(results.num_half);
results.sqrtSF = VectorXd::Zero(results.num_full);

results.sm = VectorXd::Zero(results.num_half);
results.sp = VectorXd::Zero(results.num_half);

results.phipF = VectorXd::Zero(results.num_full);
results.chipF = VectorXd::Zero(results.num_full);

results.phiF = VectorXd::Zero(results.num_full);

results.iotaF = VectorXd::Zero(results.num_full);
results.spectral_width = VectorXd::Zero(results.num_full);

results.phipH = VectorXd::Zero(results.num_half);
results.chipH = VectorXd::Zero(results.num_half);
results.bvcoH = VectorXd::Zero(results.num_half);
results.dVdsH = VectorXd::Zero(results.num_half);
results.massH = VectorXd::Zero(results.num_half);
results.presH = VectorXd::Zero(results.num_half);
results.iotaH = VectorXd::Zero(results.num_half);
results.currH = VectorXd::Zero(results.num_half);

results.rmncc = RowMatrixXd::Zero(results.num_full, s.mnsize);
results.zmnsc = RowMatrixXd::Zero(results.num_full, s.mnsize);
results.lmnsc = RowMatrixXd::Zero(results.num_full, s.mnsize);
if (s.lthreed) {
results.rmnss = RowMatrixXd::Zero(results.num_full, s.mnsize);
results.zmncs = RowMatrixXd::Zero(results.num_full, s.mnsize);
results.lmncs = RowMatrixXd::Zero(results.num_full, s.mnsize);
}
if (s.lasym) {
results.rmnsc = RowMatrixXd::Zero(results.num_full, s.mnsize);
results.zmncc = RowMatrixXd::Zero(results.num_full, s.mnsize);
results.lmncc = RowMatrixXd::Zero(results.num_full, s.mnsize);
if (s.lthreed) {
results.rmncs = RowMatrixXd::Zero(results.num_full, s.mnsize);
results.zmnss = RowMatrixXd::Zero(results.num_full, s.mnsize);
results.lmnss = RowMatrixXd::Zero(results.num_full, s.mnsize);
}
}

results.r_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.r_o = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.z_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.z_o = RowMatrixXd::Zero(results.num_full, s.nZnT);

results.ru_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.ru_o = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.zu_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.zu_o = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.rv_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.rv_o = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.zv_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.zv_o = RowMatrixXd::Zero(results.num_full, s.nZnT);

results.ruFull = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.zuFull = RowMatrixXd::Zero(results.num_full, s.nZnT);

results.r12 = RowMatrixXd::Zero(results.num_half, s.nZnT);
results.ru12 = RowMatrixXd::Zero(results.num_half, s.nZnT);
results.zu12 = RowMatrixXd::Zero(results.num_half, s.nZnT);
results.rs = RowMatrixXd::Zero(results.num_half, s.nZnT);
results.zs = RowMatrixXd::Zero(results.num_half, s.nZnT);
results.gsqrt = RowMatrixXd::Zero(results.num_half, s.nZnT);

results.guu = RowMatrixXd::Zero(results.num_half, s.nZnT);
results.guv = RowMatrixXd::Zero(results.num_half, s.nZnT);
results.gvv = RowMatrixXd::Zero(results.num_half, s.nZnT);

results.bsupu = RowMatrixXd::Zero(results.num_half, s.nZnT);
results.bsupv = RowMatrixXd::Zero(results.num_half, s.nZnT);
results.bsubu = RowMatrixXd::Zero(results.num_half, s.nZnT);
results.bsubv = RowMatrixXd::Zero(results.num_half, s.nZnT);
results.bsubvF = RowMatrixXd::Zero(results.num_full, s.nZnT);
results.total_pressure = RowMatrixXd::Zero(results.num_half, s.nZnT);

const std::size_t num_threads = radial_partitioning.size();
for (std::size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
const RadialPartitioning& r = *radial_partitioning[thread_id];
const IdealMhdModel& m = *models_from_threads[thread_id];
const RadialProfiles& p = *radial_profiles[thread_id];

const int nsMinH = r.nsMinH;
const int nsMaxH = r.nsMaxH;

const int nsMinF = r.nsMinF;
const int nsMaxFIncludingLcfs = r.nsMaxFIncludingLcfs;

const int nsMinF1 = r.nsMinF1;

for (int jH = nsMinH; jH < nsMaxH; ++jH) {

if (jH < nsMaxH - 1 || jH == fc.ns - 2) {
results.sqrtSH[jH] = p.sqrtSH[jH - nsMinH];

results.sm[jH] = p.sm[jH - nsMinH];
results.sp[jH] = p.sp[jH - nsMinH];

results.phipH[jH] = p.phipH[jH - nsMinH];
results.chipH[jH] = p.chipH[jH - nsMinH];
results.bvcoH[jH] = p.bvcoH[jH - nsMinH];
results.dVdsH[jH] = p.dVdsH[jH - nsMinH];
results.massH[jH] = p.massH[jH - nsMinH];
results.presH[jH] = p.presH[jH - nsMinH];
results.iotaH[jH] = p.iotaH[jH - nsMinH];
results.currH[jH] = p.currH[jH - nsMinH];

for (int kl = 0; kl < s.nZnT; ++kl) {
int idx_global = jH * s.nZnT + kl;
int idx_local = (jH - nsMinH) * s.nZnT + kl;

results.r12(idx_global) = m.r12[idx_local];
results.ru12(idx_global) = m.ru12[idx_local];
results.zu12(idx_global) = m.zu12[idx_local];
results.rs(idx_global) = m.rs[idx_local];
results.zs(idx_global) = m.zs[idx_local];
results.gsqrt(idx_global) = m.gsqrt[idx_local];

results.guu(idx_global) = m.guu[idx_local];
if (s.lthreed) {
results.guv(idx_global) = m.guv[idx_local];
}
results.gvv(idx_global) = m.gvv[idx_local];

results.bsupu(idx_global) = m.bsupu[idx_local];
results.bsupv(idx_global) = m.bsupv[idx_local];
results.bsubu(idx_global) = m.bsubu[idx_local];
results.bsubv(idx_global) = m.bsubv[idx_local];
results.total_pressure(idx_global) = m.totalPressure[idx_local];
}
}
}

for (int jF = nsMinF; jF < nsMaxFIncludingLcfs; ++jF) {
results.sqrtSF[jF] = p.sqrtSF[jF - nsMinF1];

results.phipF[jF] = p.phipF[jF - r.nsMinF1];

results.chipF[jF] = p.chipF[jF - r.nsMinF1];

results.iotaF[jF] = p.iotaF[jF - r.nsMinF1];
results.spectral_width[jF] = p.spectral_width[jF - r.nsMinF1];

for (int n = 0; n < s.ntor + 1; ++n) {
for (int m = 0; m < s.mpol; ++m) {

const int source_index =
((jF - nsMinF1) * s.mpol + m) * (s.ntor + 1) + n;
const int target_index = (jF * (s.ntor + 1) + n) * s.mpol + m;

results.rmncc(target_index) =
decomposed_x[thread_id]->rmncc[source_index];
results.zmnsc(target_index) =
decomposed_x[thread_id]->zmnsc[source_index];
results.lmnsc(target_index) =
decomposed_x[thread_id]->lmnsc[source_index];
if (s.lthreed) {
results.rmnss(target_index) =
decomposed_x[thread_id]->rmnss[source_index];
results.zmncs(target_index) =
decomposed_x[thread_id]->zmncs[source_index];
results.lmncs(target_index) =
decomposed_x[thread_id]->lmncs[source_index];
}
if (s.lasym) {
results.rmnsc(target_index) =
decomposed_x[thread_id]->rmnsc[source_index];
results.zmncc(target_index) =
decomposed_x[thread_id]->zmncc[source_index];
results.lmncc(target_index) =
decomposed_x[thread_id]->lmncc[source_index];
if (s.lthreed) {
results.rmncs(target_index) =
decomposed_x[thread_id]->rmncs[source_index];
results.zmnss(target_index) =
decomposed_x[thread_id]->zmnss[source_index];
results.lmnss(target_index) =
decomposed_x[thread_id]->lmnss[source_index];
}
}
}
}

double unlamscale = 1.0;
if (jF > 0) {
unlamscale = -1.0 / constants.lamscale;
}

for (int kl = 0; kl < s.nZnT; ++kl) {
int idx_global = jF * s.nZnT + kl;
int idx_local = (jF - nsMinF) * s.nZnT + kl;
int idx_local1 = (jF - nsMinF1) * s.nZnT + kl;

double blmn_e_full = m.blmn_e[idx_local];
if (s.lasym) {

const int l = kl % s.nThetaEff;
if (l < s.nThetaReduced) {
blmn_e_full += m.blmn_asym_e[idx_local];
}
}
results.bsubvF(idx_global) = blmn_e_full * unlamscale;

results.r_e(idx_global) = m.r1_e[idx_local1];
results.r_o(idx_global) = m.r1_o[idx_local1];
results.z_e(idx_global) = m.z1_e[idx_local1];
results.z_o(idx_global) = m.z1_o[idx_local1];
results.ru_e(idx_global) = m.ru_e[idx_local1];
results.ru_o(idx_global) = m.ru_o[idx_local1];
results.zu_e(idx_global) = m.zu_e[idx_local1];
results.zu_o(idx_global) = m.zu_o[idx_local1];
if (s.lthreed) {
results.rv_e(idx_global) = m.rv_e[idx_local1];
results.rv_o(idx_global) = m.rv_o[idx_local1];
results.zv_e(idx_global) = m.zv_e[idx_local1];
results.zv_o(idx_global) = m.zv_o[idx_local1];
} else {
results.rv_e(idx_global) = 0.0;
results.rv_o(idx_global) = 0.0;
results.zv_e(idx_global) = 0.0;
results.zv_o(idx_global) = 0.0;
}

results.ruFull(idx_global) = m.ruFull[idx_local];
results.zuFull(idx_global) = m.zuFull[idx_local];
}
}

if (thread_id == 0) {

results.currv = radial_profiles[thread_id]->currv;
}
}

return results;
}

void vmecpp::MeshBledingBSubZeta(const Sizes& s, const FlowControl& fc,
VmecInternalResults& m_vmec_internal_results) {

for (int jH = fc.ns - 3; jH >= 0; jH--) {

int jF = jH + 1;
for (int kl = 0; kl < s.nZnT; ++kl) {
int idx_sourceH = (jH + 1) * s.nZnT + kl;
int idx_sourceF = jF * s.nZnT + kl;
int idx_targetH = jH * s.nZnT + kl;

m_vmec_internal_results.bsubv(idx_targetH) =
2.0 * m_vmec_internal_results.bsubvF(idx_sourceF) -
m_vmec_internal_results.bsubv(idx_sourceH);
}
}
}

vmecpp::PoloidalCurrentToFixBSubV vmecpp::ComputePoloidalCurrentToFixBSubV(
const Sizes& s, const VmecInternalResults& vmec_internal_results) {
PoloidalCurrentToFixBSubV poloidal_current_to_fix_bsubv;

poloidal_current_to_fix_bsubv.poloidal_current_deviation =
VectorXd::Zero(vmec_internal_results.num_half);

for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
double poloidal_current_from_bsubv = 0.0;
for (int kl = 0; kl < s.nZnT; ++kl) {
int iHalf = jH * s.nZnT + kl;
int l = kl % s.nThetaEff;
poloidal_current_from_bsubv +=
vmec_internal_results.bsubv(iHalf) * s.wInt[l];
}

poloidal_current_to_fix_bsubv.poloidal_current_deviation[jH] =
poloidal_current_from_bsubv - vmec_internal_results.bvcoH[jH];
}

return poloidal_current_to_fix_bsubv;
}

void vmecpp::FixupPoloidalCurrent(
const Sizes& s,
const PoloidalCurrentToFixBSubV& poloidal_current_to_fix_bsubv,
VmecInternalResults& m_vmec_internal_results) {
for (int jH = 0; jH < m_vmec_internal_results.num_half; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
int iHalf = jH * s.nZnT + kl;

m_vmec_internal_results.bsubv(iHalf) -=
poloidal_current_to_fix_bsubv.poloidal_current_deviation[jH];
}
}
}

void vmecpp::RecomputeToroidalFlux(
const FlowControl& fc, VmecInternalResults& m_vmec_internal_results) {

m_vmec_internal_results.phiF[0] = 0.0;
for (int jF = 1; jF < fc.ns; ++jF) {
m_vmec_internal_results.phiF[jF] = m_vmec_internal_results.phiF[jF - 1] +
m_vmec_internal_results.phipH[jF - 1];
}

const double scaling_factor =
m_vmec_internal_results.sign_of_jacobian * 2.0 * M_PI * fc.deltaS;
for (int jF = 0; jF < fc.ns; ++jF) {
m_vmec_internal_results.phiF[jF] *= scaling_factor;
}
}

vmecpp::RemainingMetric vmecpp::ComputeRemainingMetric(
const Sizes& s, const VmecInternalResults& vmec_internal_results) {
RemainingMetric remaining_metric;

remaining_metric.rv12 =
RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
remaining_metric.zv12 =
RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
remaining_metric.rs12 =
RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
remaining_metric.zs12 =
RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
remaining_metric.gsu =
RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
remaining_metric.gsv =
RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
int idxH = jH * s.nZnT + kl;
int idxFi = idxH;
int idxFo = (jH + 1) * s.nZnT + kl;

double rv_e12 = (vmec_internal_results.rv_e(idxFo) +
vmec_internal_results.rv_e(idxFi)) /
2.0;
double rv_o12 = (vmec_internal_results.rv_o(idxFo) +
vmec_internal_results.rv_o(idxFi)) /
2.0;
remaining_metric.rv12(idxH) =
rv_e12 + vmec_internal_results.sqrtSH[jH] * rv_o12;

double zv_e12 = (vmec_internal_results.zv_e(idxFo) +
vmec_internal_results.zv_e(idxFi)) /
2.0;
double zv_o12 = (vmec_internal_results.zv_o(idxFo) +
vmec_internal_results.zv_o(idxFi)) /
2.0;
remaining_metric.zv12(idxH) =
zv_e12 + vmec_internal_results.sqrtSH[jH] * zv_o12;

double r_o12 = (vmec_internal_results.r_o(idxFo) +
vmec_internal_results.r_o(idxFi)) /
2.0;
double z_o12 = (vmec_internal_results.z_o(idxFo) +
vmec_internal_results.z_o(idxFi)) /
2.0;
remaining_metric.rs12(idxH) =
vmec_internal_results.rs(idxH) +
r_o12 / (2.0 * vmec_internal_results.sqrtSH[jH]);
remaining_metric.zs12(idxH) =
vmec_internal_results.zs(idxH) +
z_o12 / (2.0 * vmec_internal_results.sqrtSH[jH]);

remaining_metric.gsu(idxH) =
remaining_metric.rs12(idxH) * vmec_internal_results.ru12(idxH) +
remaining_metric.zs12(idxH) * vmec_internal_results.zu12(idxH);
remaining_metric.gsv(idxH) =
remaining_metric.rs12(idxH) * remaining_metric.rv12(idxH) +
remaining_metric.zs12(idxH) * remaining_metric.zv12(idxH);
}
}

return remaining_metric;
}

vmecpp::CylindricalComponentsOfB vmecpp::BCylindricalComponents(
const Sizes& s, const VmecInternalResults& vmec_internal_results,
const RemainingMetric& remaining_metric) {
CylindricalComponentsOfB b_cylindrical;

b_cylindrical.b_r = RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
b_cylindrical.b_phi =
RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
b_cylindrical.b_z = RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);

for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
int idxH = jH * s.nZnT + kl;

b_cylindrical.b_r(idxH) =
vmec_internal_results.bsupu(idxH) * vmec_internal_results.ru12(idxH) +
vmec_internal_results.bsupv(idxH) * remaining_metric.rv12(idxH);
b_cylindrical.b_phi(idxH) =
vmec_internal_results.bsupv(idxH) * vmec_internal_results.r12(idxH);
b_cylindrical.b_z(idxH) =
vmec_internal_results.bsupu(idxH) * vmec_internal_results.zu12(idxH) +
vmec_internal_results.bsupv(idxH) * remaining_metric.zv12(idxH);
}
}

return b_cylindrical;
}

vmecpp::BSubSHalf vmecpp::ComputeBSubSOnHalfGrid(
const Sizes& s, const VmecInternalResults& vmec_internal_results,
const RemainingMetric& remaining_metric) {
BSubSHalf bsubs_half;
bsubs_half.bsubs_half =
RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);

for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
int idxH = jH * s.nZnT + kl;
bsubs_half.bsubs_half(idxH) =
vmec_internal_results.bsupu(idxH) * remaining_metric.gsu(idxH) +
vmec_internal_results.bsupv(idxH) * remaining_metric.gsv(idxH);
}
}

return bsubs_half;
}

vmecpp::BSubSFull vmecpp::PutBSubSOnFullGrid(
const Sizes& s, const VmecInternalResults& vmec_internal_results,
const BSubSHalf& bsubs_half) {
BSubSFull bsubs_full;
bsubs_full.bsubs_full =
RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

for (int jF = 1; jF < vmec_internal_results.num_full - 1; ++jF) {
const int jHi = jF - 1;
const int jHo = jF;
for (int kl = 0; kl < s.nZnT; ++kl) {
bsubs_full.bsubs_full(jF * s.nZnT + kl) =
(bsubs_half.bsubs_half(jHo * s.nZnT + kl) +
bsubs_half.bsubs_half(jHi * s.nZnT + kl)) /
2.0;
}
}

return bsubs_full;
}

vmecpp::SymmetryDecomposedCovariantB vmecpp::DecomposeCovariantBBySymmetry(
const Sizes& s, const VmecInternalResults& vmec_internal_results,
const BSubSFull& bsubs_full) {
SymmetryDecomposedCovariantB decomposed_bcov;

decomposed_bcov.bsubs_s = RowMatrixXd::Zero(
vmec_internal_results.num_full, vmec_internal_results.nZnT_reduced);
decomposed_bcov.bsubu_s = RowMatrixXd::Zero(
vmec_internal_results.num_half, vmec_internal_results.nZnT_reduced);
decomposed_bcov.bsubv_s = RowMatrixXd::Zero(
vmec_internal_results.num_half, vmec_internal_results.nZnT_reduced);

if (s.lasym) {
decomposed_bcov.bsubs_a = RowMatrixXd::Zero(
vmec_internal_results.num_full, vmec_internal_results.nZnT_reduced);
decomposed_bcov.bsubu_a = RowMatrixXd::Zero(
vmec_internal_results.num_half, vmec_internal_results.nZnT_reduced);
decomposed_bcov.bsubv_a = RowMatrixXd::Zero(
vmec_internal_results.num_half, vmec_internal_results.nZnT_reduced);

for (int jF = 0; jF < vmec_internal_results.num_full; ++jF) {
for (int kl = 0; kl < vmec_internal_results.nZnT_reduced; ++kl) {
const int k = kl / s.nThetaReduced;
const int l = kl % s.nThetaReduced;

const int source_index = jF * s.nZnT + (k * s.nThetaEven + l);

const int l_reversed = (s.nThetaEven - l) % s.nThetaEven;
const int k_reversed = (s.nZeta - k) % s.nZeta;
const int kl_reversed = k_reversed * s.nThetaEven + l_reversed;

const int source_index_reversed = jF * s.nZnT + kl_reversed;

const int target_index = jF * vmec_internal_results.nZnT_reduced + kl;

decomposed_bcov.bsubs_s(target_index) =
0.5 * (bsubs_full.bsubs_full(source_index) -
bsubs_full.bsubs_full(source_index_reversed));
decomposed_bcov.bsubs_a(target_index) =
0.5 * (bsubs_full.bsubs_full(source_index) +
bsubs_full.bsubs_full(source_index_reversed));
}
}
for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
for (int kl = 0; kl < vmec_internal_results.nZnT_reduced; ++kl) {
const int k = kl / s.nThetaReduced;
const int l = kl % s.nThetaReduced;

const int source_index = jH * s.nZnT + (k * s.nThetaEven + l);

const int l_reversed = (s.nThetaEven - l) % s.nThetaEven;
const int k_reversed = (s.nZeta - k) % s.nZeta;
const int kl_reversed = k_reversed * s.nThetaEven + l_reversed;

const int source_index_reversed = jH * s.nZnT + kl_reversed;

const int target_index = jH * vmec_internal_results.nZnT_reduced + kl;

decomposed_bcov.bsubu_s(target_index) =
0.5 * (vmec_internal_results.bsubu(source_index) +
vmec_internal_results.bsubu(source_index_reversed));
decomposed_bcov.bsubu_a(target_index) =
0.5 * (vmec_internal_results.bsubu(source_index) -
vmec_internal_results.bsubu(source_index_reversed));
decomposed_bcov.bsubv_s(target_index) =
0.5 * (vmec_internal_results.bsubv(source_index) +
vmec_internal_results.bsubv(source_index_reversed));
decomposed_bcov.bsubv_a(target_index) =
0.5 * (vmec_internal_results.bsubv(source_index) -
vmec_internal_results.bsubv(source_index_reversed));
}
}
} else {

for (int jF = 0; jF < vmec_internal_results.num_full; ++jF) {
for (int kl = 0; kl < vmec_internal_results.nZnT_reduced; ++kl) {
const int source_index = jF * s.nZnT + kl;
const int target_index = jF * vmec_internal_results.nZnT_reduced + kl;
decomposed_bcov.bsubs_s(target_index) =
bsubs_full.bsubs_full(source_index);
}
}
for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
const int source_index = jH * s.nZnT + kl;
const int target_index = jH * vmec_internal_results.nZnT_reduced + kl;
decomposed_bcov.bsubu_s(target_index) =
vmec_internal_results.bsubu(source_index);
decomposed_bcov.bsubv_s(target_index) =
vmec_internal_results.bsubv(source_index);
}
}
}

return decomposed_bcov;
}

vmecpp::CovariantBDerivatives vmecpp::LowPassFilterCovariantB(
const Sizes& s, const FourierBasisFastPoloidal& t,
const SymmetryDecomposedCovariantB& decomposed_bcov,
VmecInternalResults& m_vmec_internal_results) {
CovariantBDerivatives covariant_b_derivatives;

covariant_b_derivatives.bsubsu =
RowMatrixXd::Zero(m_vmec_internal_results.num_full, s.nZnT);
covariant_b_derivatives.bsubsv =
RowMatrixXd::Zero(m_vmec_internal_results.num_full, s.nZnT);

covariant_b_derivatives.bsubuv =
RowMatrixXd::Zero(m_vmec_internal_results.num_half, s.nZnT);
covariant_b_derivatives.bsubvu =
RowMatrixXd::Zero(m_vmec_internal_results.num_half, s.nZnT);

std::vector<double> bsubsu_s(m_vmec_internal_results.num_full * s.nZnT, 0.0);
std::vector<double> bsubsv_s(m_vmec_internal_results.num_full * s.nZnT, 0.0);
std::vector<double> bsubsu_a;
std::vector<double> bsubsv_a;
if (s.lasym) {
bsubsu_a.resize(m_vmec_internal_results.num_full * s.nZnT);
bsubsv_a.resize(m_vmec_internal_results.num_full * s.nZnT);
}

std::vector<double> bsubu_filtered_s(
m_vmec_internal_results.num_half * s.nZnT, 0.0);
std::vector<double> bsubv_filtered_s(
m_vmec_internal_results.num_half * s.nZnT, 0.0);
std::vector<double> bsubu_filtered_a;
std::vector<double> bsubv_filtered_a;
if (s.lasym) {
bsubu_filtered_a.resize(m_vmec_internal_results.num_half * s.nZnT);
bsubv_filtered_a.resize(m_vmec_internal_results.num_half * s.nZnT);
}

std::vector<double> bsubvu_s(m_vmec_internal_results.num_half * s.nZnT, 0.0);
std::vector<double> bsubuv_s(m_vmec_internal_results.num_half * s.nZnT, 0.0);
std::vector<double> bsubvu_a;
std::vector<double> bsubuv_a;
if (s.lasym) {
bsubvu_a.resize(m_vmec_internal_results.num_half * s.nZnT);
bsubuv_a.resize(m_vmec_internal_results.num_half * s.nZnT);
}

for (int jF = 0; jF < m_vmec_internal_results.num_full; ++jF) {
for (int m = 0; m < s.mpol; ++m) {
for (int n = 0; n <= s.ntor; ++n) {

double dnorm1 = 1.0;
if (m == s.mnyq) {
dnorm1 /= 2.0;
}
if (n == s.nnyq && n != 0) {
dnorm1 /= 2.0;
}

double bsubsmn1 = 0.0;
double bsubsmn2 = 0.0;

double bsubsmn3 = 0.0;
double bsubsmn4 = 0.0;

for (int k = 0; k < s.nZeta; ++k) {

const int idx_kn = k * (s.nnyq2 + 1) + n;
for (int l = 0; l < s.nThetaReduced; ++l) {
const int idx_ml = m * s.nThetaReduced + l;
const int kl = k * s.nThetaReduced + l;

const int source_index =
jF * m_vmec_internal_results.nZnT_reduced + kl;

const double tsini1 = t.sinmui[idx_ml] * t.cosnv[idx_kn] * dnorm1;

const double tsini2 = t.cosmui[idx_ml] * t.sinnv[idx_kn] * dnorm1;

bsubsmn1 += tsini1 * decomposed_bcov.bsubs_s(source_index);

bsubsmn2 += tsini2 * decomposed_bcov.bsubs_s(source_index);

if (s.lasym) {

const double tcosi1 = t.cosmui[idx_ml] * t.cosnv[idx_kn] * dnorm1;

const double tcosi2 = t.sinmui[idx_ml] * t.sinnv[idx_kn] * dnorm1;

bsubsmn3 += tcosi1 * decomposed_bcov.bsubs_a(source_index);

bsubsmn4 += tcosi2 * decomposed_bcov.bsubs_a(source_index);
}
}
}

for (int k = 0; k < s.nZeta; ++k) {

const int idx_kn = k * (s.nnyq2 + 1) + n;
for (int l = 0; l < s.nThetaReduced; ++l) {
const int idx_ml = m * s.nThetaReduced + l;

const int kl = k * s.nThetaReduced + l;
const int target_index =
jF * m_vmec_internal_results.nZnT_reduced + kl;

const double tcosm1 = t.cosmum[idx_ml] * t.cosnv[idx_kn];
const double tcosm2 = t.sinmum[idx_ml] * t.sinnv[idx_kn];
bsubsu_s[target_index] += tcosm1 * bsubsmn1 + tcosm2 * bsubsmn2;

const double tcosn1 = t.sinmu[idx_ml] * t.sinnvn[idx_kn];
const double tcosn2 = t.cosmu[idx_ml] * t.cosnvn[idx_kn];
bsubsv_s[target_index] += tcosn1 * bsubsmn1 + tcosn2 * bsubsmn2;

if (s.lasym) {
const double tsinm1 = t.sinmum[idx_ml] * t.cosnv[idx_kn];
const double tsinm2 = t.cosmum[idx_ml] * t.sinnv[idx_kn];
bsubsu_a[target_index] += tsinm1 * bsubsmn3 + tsinm2 * bsubsmn4;

const double tsinn1 = t.cosmu[idx_ml] * t.sinnvn[idx_kn];
const double tsinn2 = t.sinmu[idx_ml] * t.cosnvn[idx_kn];
bsubsv_a[target_index] += tsinn1 * bsubsmn3 + tsinn2 * bsubsmn4;
}
}
}
}
}
}

for (int jH = 0; jH < m_vmec_internal_results.num_half; ++jH) {
for (int m = 0; m < s.mpol; ++m) {
for (int n = 0; n <= s.ntor; ++n) {

double dnorm1 = 1.0;
if (m == s.mnyq) {
dnorm1 /= 2.0;
}
if (n == s.nnyq && n != 0) {
dnorm1 /= 2.0;
}

double bsubumn1 = 0.0;
double bsubumn2 = 0.0;
double bsubvmn1 = 0.0;
double bsubvmn2 = 0.0;

double bsubumn3 = 0.0;
double bsubumn4 = 0.0;
double bsubvmn3 = 0.0;
double bsubvmn4 = 0.0;

for (int k = 0; k < s.nZeta; ++k) {

const int idx_kn = k * (s.nnyq2 + 1) + n;
for (int l = 0; l < s.nThetaReduced; ++l) {
const int idx_ml = m * s.nThetaReduced + l;

const int kl = k * s.nThetaReduced + l;
const int source_index =
jH * m_vmec_internal_results.nZnT_reduced + kl;

const double tcosi1 = t.cosmui[idx_ml] * t.cosnv[idx_kn] * dnorm1;

const double tcosi2 = t.sinmui[idx_ml] * t.sinnv[idx_kn] * dnorm1;

bsubvmn1 += tcosi1 * decomposed_bcov.bsubv_s(source_index);

bsubvmn2 += tcosi2 * decomposed_bcov.bsubv_s(source_index);

bsubumn1 += tcosi1 * decomposed_bcov.bsubu_s(source_index);

bsubumn2 += tcosi2 * decomposed_bcov.bsubu_s(source_index);

if (s.lasym) {

const double tsini1 = t.sinmui[idx_ml] * t.cosnv[idx_kn] * dnorm1;

const double tsini2 = t.cosmui[idx_ml] * t.sinnv[idx_kn] * dnorm1;

bsubvmn3 += tsini1 * decomposed_bcov.bsubv_a(source_index);

bsubvmn4 += tsini2 * decomposed_bcov.bsubv_a(source_index);

bsubumn3 += tsini1 * decomposed_bcov.bsubu_a(source_index);

bsubumn4 += tsini2 * decomposed_bcov.bsubu_a(source_index);
}
}
}

for (int k = 0; k < s.nZeta; ++k) {
const int idx_kn = k * (s.nnyq2 + 1) + n;
for (int l = 0; l < s.nThetaReduced; ++l) {
const int idx_ml = m * s.nThetaReduced + l;

const int kl = k * s.nThetaReduced + l;
const int target_index =
jH * m_vmec_internal_results.nZnT_reduced + kl;

const double tcos1 = t.cosmu[idx_ml] * t.cosnv[idx_kn];
const double tcos2 = t.sinmu[idx_ml] * t.sinnv[idx_kn];
bsubu_filtered_s[target_index] +=
tcos1 * bsubumn1 + tcos2 * bsubumn2;
bsubv_filtered_s[target_index] +=
tcos1 * bsubvmn1 + tcos2 * bsubvmn2;

const double tsinm1 = t.sinmum[idx_ml] * t.cosnv[idx_kn];
const double tsinm2 = t.cosmum[idx_ml] * t.sinnv[idx_kn];
bsubvu_s[target_index] += tsinm1 * bsubvmn1 + tsinm2 * bsubvmn2;

const double tsinn1 = t.cosmu[idx_ml] * t.sinnvn[idx_kn];
const double tsinn2 = t.sinmu[idx_ml] * t.cosnvn[idx_kn];
bsubuv_s[target_index] += tsinn1 * bsubumn1 + tsinn2 * bsubumn2;

if (s.lasym) {
const double tsin1 = t.sinmu[idx_ml] * t.cosnv[idx_kn];
const double tsin2 = t.cosmu[idx_ml] * t.sinnv[idx_kn];
bsubu_filtered_a[target_index] +=
tsin1 * bsubumn3 + tsin2 * bsubumn4;
bsubv_filtered_a[target_index] +=
tsin1 * bsubvmn3 + tsin2 * bsubvmn4;

const double tcosm1 = t.cosmum[idx_ml] * t.cosnv[idx_kn];
const double tcosm2 = t.sinmum[idx_ml] * t.sinnv[idx_kn];
bsubvu_a[target_index] += tcosm1 * bsubvmn3 + tcosm2 * bsubvmn4;

const double tcosn1 = t.sinmu[idx_ml] * t.sinnvn[idx_kn];
const double tcosn2 = t.cosmu[idx_ml] * t.cosnvn[idx_kn];
bsubuv_a[target_index] += tcosn1 * bsubumn3 + tcosn2 * bsubumn4;
}
}
}
}
}
}

if (s.lasym) {

const int nZnT_reduced = m_vmec_internal_results.nZnT_reduced;
for (int jH = 0; jH < m_vmec_internal_results.num_half; ++jH) {
for (int k = 0; k < s.nZeta; ++k) {
const int k_reversed = (s.nZeta - k) % s.nZeta;
for (int l = 0; l < s.nThetaReduced; ++l) {
const int source_index =
jH * nZnT_reduced + (k * s.nThetaReduced + l);
const int target_index = jH * s.nZnT + (k * s.nThetaEff + l);

m_vmec_internal_results.bsubu(target_index) =
bsubu_filtered_s[source_index] + bsubu_filtered_a[source_index];
m_vmec_internal_results.bsubv(target_index) =
bsubv_filtered_s[source_index] + bsubv_filtered_a[source_index];
}
for (int l = s.nThetaReduced; l < s.nThetaEven; ++l) {
const int l_reversed = (s.nThetaEven - l) % s.nThetaEven;
const int source_index_reversed =
jH * nZnT_reduced + (k_reversed * s.nThetaReduced + l_reversed);
const int target_index = jH * s.nZnT + (k * s.nThetaEff + l);

m_vmec_internal_results.bsubu(target_index) =
bsubu_filtered_s[source_index_reversed] -
bsubu_filtered_a[source_index_reversed];
m_vmec_internal_results.bsubv(target_index) =
bsubv_filtered_s[source_index_reversed] -
bsubv_filtered_a[source_index_reversed];
}
}
}
} else {

for (int jH = 0; jH < m_vmec_internal_results.num_half; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
const int idx_kl = jH * s.nZnT + kl;
m_vmec_internal_results.bsubu(idx_kl) = bsubu_filtered_s[idx_kl];
m_vmec_internal_results.bsubv(idx_kl) = bsubv_filtered_s[idx_kl];
}
}
}

if (s.lasym) {
const int nZnT_reduced = m_vmec_internal_results.nZnT_reduced;
for (int jH = 0; jH < m_vmec_internal_results.num_half; ++jH) {
for (int k = 0; k < s.nZeta; ++k) {
const int k_reversed = (s.nZeta - k) % s.nZeta;
for (int l = 0; l < s.nThetaReduced; ++l) {
const int source_index =
jH * nZnT_reduced + (k * s.nThetaReduced + l);
const int target_index = jH * s.nZnT + (k * s.nThetaEff + l);

covariant_b_derivatives.bsubvu(target_index) =
bsubvu_s[source_index] + bsubvu_a[source_index];
covariant_b_derivatives.bsubuv(target_index) =
bsubuv_s[source_index] + bsubuv_a[source_index];
}
for (int l = s.nThetaReduced; l < s.nThetaEven; ++l) {
const int l_reversed = (s.nThetaEven - l) % s.nThetaEven;
const int source_index_reversed =
jH * nZnT_reduced + (k_reversed * s.nThetaReduced + l_reversed);
const int target_index = jH * s.nZnT + (k * s.nThetaEff + l);

covariant_b_derivatives.bsubvu(target_index) =
-bsubvu_s[source_index_reversed] +
bsubvu_a[source_index_reversed];
covariant_b_derivatives.bsubuv(target_index) =
-bsubuv_s[source_index_reversed] +
bsubuv_a[source_index_reversed];
}
}
}
} else {
for (int jH = 0; jH < m_vmec_internal_results.num_half; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
const int idx_kl = jH * s.nZnT + kl;
covariant_b_derivatives.bsubvu(idx_kl) = bsubvu_s[idx_kl];
covariant_b_derivatives.bsubuv(idx_kl) = bsubuv_s[idx_kl];
}
}
}

if (s.lasym) {

const int nZnT_reduced = m_vmec_internal_results.nZnT_reduced;
for (int jF = 0; jF < m_vmec_internal_results.num_full; ++jF) {
for (int k = 0; k < s.nZeta; ++k) {
const int k_reversed = (s.nZeta - k) % s.nZeta;
for (int l = s.nThetaReduced; l < s.nThetaEven; ++l) {
const int l_reversed = (s.nThetaEven - l) % s.nThetaEven;
const int source_index_reversed =
jF * nZnT_reduced + (k_reversed * s.nThetaReduced + l_reversed);
const int target_index = jF * s.nZnT + (k * s.nThetaEff + l);

covariant_b_derivatives.bsubsu(target_index) =
bsubsu_s[source_index_reversed] - bsubsu_a[source_index_reversed];
covariant_b_derivatives.bsubsv(target_index) =
bsubsv_s[source_index_reversed] - bsubsv_a[source_index_reversed];
}
for (int l = 0; l < s.nThetaReduced; ++l) {
const int source_index =
jF * nZnT_reduced + (k * s.nThetaReduced + l);
const int target_index = jF * s.nZnT + (k * s.nThetaEff + l);

covariant_b_derivatives.bsubsu(target_index) =
bsubsu_s[source_index] + bsubsu_a[source_index];
covariant_b_derivatives.bsubsv(target_index) =
bsubsv_s[source_index] + bsubsv_a[source_index];
}
}
}
} else {

for (int jF = 0; jF < m_vmec_internal_results.num_full; ++jF) {
for (int kl = 0; kl < s.nZnT; ++kl) {
const int idx_kl = jF * s.nZnT + kl;
covariant_b_derivatives.bsubsu(idx_kl) = bsubsu_s[idx_kl];
covariant_b_derivatives.bsubsv(idx_kl) = bsubsv_s[idx_kl];
}
}
}

return covariant_b_derivatives;
}

void vmecpp::ExtrapolateBSubS(const Sizes& s, const FlowControl& fc,
BSubSFull& m_bsubs_full) {
for (int kl = 0; kl < s.nZnT; ++kl) {

const int index_0 = 0 * s.nZnT + kl;
const int index_1 = 1 * s.nZnT + kl;
const int index_2 = 2 * s.nZnT + kl;
m_bsubs_full.bsubs_full(index_0) = 2.0 * m_bsubs_full.bsubs_full(index_1) -
m_bsubs_full.bsubs_full(index_2);

const int index_ns_1 = (fc.ns - 1) * s.nZnT + kl;
const int index_ns_2 = (fc.ns - 2) * s.nZnT + kl;
const int index_ns_3 = (fc.ns - 3) * s.nZnT + kl;
m_bsubs_full.bsubs_full(index_ns_1) =
2.0 * m_bsubs_full.bsubs_full(index_ns_2) -
m_bsubs_full.bsubs_full(index_ns_3);
}
}

vmecpp::JxBOutFileContents vmecpp::ComputeJxBOutputFileContents(
const Sizes& s, const FlowControl& fc,
const VmecInternalResults& vmec_internal_results,
const BSubSFull& bsubs_full,
const CovariantBDerivatives& covariant_b_derivatives,
const bool return_outputs_even_if_not_converged, VmecStatus vmec_status) {
JxBOutFileContents jxbout;

jxbout.itheta = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
jxbout.izeta = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
jxbout.bdotk = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

jxbout.amaxfor = VectorXd::Zero(vmec_internal_results.num_full);
jxbout.aminfor = VectorXd::Zero(vmec_internal_results.num_full);
jxbout.avforce = VectorXd::Zero(vmec_internal_results.num_full);
jxbout.pprim = VectorXd::Zero(vmec_internal_results.num_full);
jxbout.jdotb = VectorXd::Zero(vmec_internal_results.num_full);
jxbout.bdotb = VectorXd::Zero(vmec_internal_results.num_full);
jxbout.bdotgradv = VectorXd::Zero(vmec_internal_results.num_full);
jxbout.jpar2 = VectorXd::Zero(vmec_internal_results.num_full);
jxbout.jperp2 = VectorXd::Zero(vmec_internal_results.num_full);
jxbout.phin = VectorXd::Zero(vmec_internal_results.num_full);

jxbout.phin[fc.ns - 1] = 1.0;

jxbout.jsupu3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
jxbout.jsupv3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
jxbout.jsups3 = RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);

jxbout.bsupu3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
jxbout.bsupv3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

jxbout.jcrossb = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
jxbout.jxb_gradp = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
jxbout.jdotb_sqrtg =
RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

jxbout.sqrtg3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

jxbout.bsubu3 = RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
jxbout.bsubv3 = RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
jxbout.bsubs3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

std::vector<double> pprim(fc.ns, 0.0);

std::vector<double> sqgb2(s.nZnT, 0.0);

std::vector<double> kperpu(s.nZnT, 0.0);
std::vector<double> kperpv(s.nZnT, 0.0);

std::vector<double> kp2(s.nZnT, 0.0);

std::vector<double> sqrtg(s.nZnT, 0.0);

std::vector<double> bsupu1(s.nZnT, 0.0);
std::vector<double> bsupv1(s.nZnT, 0.0);

std::vector<double> bsubu1(s.nZnT, 0.0);
std::vector<double> bsubv1(s.nZnT, 0.0);

std::vector<double> jxb(s.nZnT, 0.0);

static constexpr double dnorm1 = 4.0 * M_PI * M_PI;

for (int jF = 1; jF < fc.ns - 1; ++jF) {
const int jHi = jF - 1;
const int jHo = jF;

const double ovp =
2.0 /
(vmec_internal_results.dVdsH[jHo] + vmec_internal_results.dVdsH[jHi]) /
dnorm1;

const double tjnorm = ovp * vmec_internal_results.sign_of_jacobian;

double pprime =
1.0 / MU_0 *
(vmec_internal_results.presH[jHo] - vmec_internal_results.presH[jHi]) /
fc.deltaS;

const double pprime_ovp = pprime * ovp;
const double pnorm = 1.0 / (std::abs(pprime_ovp) + DBL_EPSILON);

double force_residual_max = -DBL_MAX;
double force_residual_min = DBL_MAX;

double average_force_residual = 0.0;
double average_pressure_gradient = 0.0;

double average_jdotb = 0.0;
double average_bdotb = 0.0;

double average_jpar2 = 0.0;
double average_jperp2 = 0.0;

for (int kl = 0; kl < s.nZnT; ++kl) {
const int l = kl % s.nThetaEff;
const int target_index = jF * s.nZnT + kl;

const double gsqrt_outside =
vmec_internal_results.gsqrt(jHo * s.nZnT + kl);
const double gsqrt_inside =
vmec_internal_results.gsqrt(jHi * s.nZnT + kl);

const double bsq_outside =
vmec_internal_results.total_pressure(jHo * s.nZnT + kl) -
vmec_internal_results.presH[jHo];
const double bsq_inside =
vmec_internal_results.total_pressure(jHi * s.nZnT + kl) -
vmec_internal_results.presH[jHi];

sqgb2[kl] = gsqrt_outside * bsq_outside + gsqrt_inside * bsq_inside;

const double bsubu_outside =
vmec_internal_results.bsubu(jHo * s.nZnT + kl);
const double bsubu_inside =
vmec_internal_results.bsubu(jHi * s.nZnT + kl);

const double bsubv_outside =
vmec_internal_results.bsubv(jHo * s.nZnT + kl);
const double bsubv_inside =
vmec_internal_results.bsubv(jHi * s.nZnT + kl);

kperpu[kl] = 0.5 * (bsubv_outside + bsubv_inside) * pprime / sqgb2[kl];

kperpv[kl] = -0.5 * (bsubu_outside + bsubu_inside) * pprime / sqgb2[kl];

const double guu_o = vmec_internal_results.guu(jHo * s.nZnT + kl);
const double guu_i = vmec_internal_results.guu(jHi * s.nZnT + kl);

const double guv_o = vmec_internal_results.guv(jHo * s.nZnT + kl);
const double guv_i = vmec_internal_results.guv(jHi * s.nZnT + kl);

const double gvv_o = vmec_internal_results.gvv(jHo * s.nZnT + kl);
const double gvv_i = vmec_internal_results.gvv(jHi * s.nZnT + kl);

const double term_uu = kperpu[kl] * kperpu[kl] * (guu_o + guu_i) / 2.0;
const double term_uv = kperpu[kl] * kperpv[kl] * (guv_o + guv_i) / 2.0;
const double term_vv = kperpv[kl] * kperpv[kl] * (gvv_o + gvv_i) / 2.0;

kp2[kl] = term_uu + 2.0 * term_uv + term_vv;

const double bsubsu = covariant_b_derivatives.bsubsu(jF * s.nZnT + kl);
const double bsubsv = covariant_b_derivatives.bsubsv(jF * s.nZnT + kl);

const double bsubus = (bsubu_outside - bsubu_inside) / fc.deltaS;
const double bsubvs = (bsubv_outside - bsubv_inside) / fc.deltaS;

jxbout.itheta(target_index) = (bsubsv - bsubvs) / MU_0;
jxbout.izeta(target_index) = (-bsubsu + bsubus) / MU_0;

sqrtg[kl] = (gsqrt_outside + gsqrt_inside) / 2.0;

const double bsupu_outside =
vmec_internal_results.bsupu(jHo * s.nZnT + kl);
const double bsupu_inside =
vmec_internal_results.bsupu(jHi * s.nZnT + kl);

const double bsupv_outside =
vmec_internal_results.bsupv(jHo * s.nZnT + kl);
const double bsupv_inside =
vmec_internal_results.bsupv(jHi * s.nZnT + kl);

bsupu1[kl] =
(bsupu_outside * gsqrt_outside + bsupu_inside * gsqrt_inside) /
(2.0 * sqrtg[kl]);
bsupv1[kl] =
(bsupv_outside * gsqrt_outside + bsupv_inside * gsqrt_inside) /
(2.0 * sqrtg[kl]);

bsubu1[kl] = (bsubu_outside + bsubu_inside) / 2.0;
bsubv1[kl] = (bsubv_outside + bsubv_inside) / 2.0;

jxb[kl] = ovp * (jxbout.itheta(target_index) * bsupv1[kl] -
jxbout.izeta(target_index) * bsupu1[kl]);

jxbout.bdotk(target_index) = jxbout.itheta(target_index) * bsubu1[kl] +
jxbout.izeta(target_index) * bsubv1[kl];

const double force_residual = jxb[kl] - pprime_ovp;

force_residual_max = std::max(force_residual_max, force_residual * pnorm);
force_residual_min = std::min(force_residual_min, force_residual * pnorm);

average_force_residual += force_residual * s.wInt[l];
average_pressure_gradient += pprime_ovp * s.wInt[l];

average_jdotb += jxbout.bdotk(target_index) * s.wInt[l];
average_bdotb += sqgb2[kl] * s.wInt[l];

average_jpar2 += jxbout.bdotk(target_index) * jxbout.bdotk(target_index) *
s.wInt[l] / sqgb2[kl];
average_jperp2 += kp2[kl] * sqrtg[kl] * s.wInt[l];
}

jxbout.amaxfor[jF] = 100.0 * std::min(force_residual_max, 9.999);
jxbout.aminfor[jF] = 100.0 * std::max(force_residual_min, -9.999);

jxbout.avforce[jF] = average_force_residual;
jxbout.pprim[jF] = average_pressure_gradient;

jxbout.jdotb[jF] = dnorm1 * tjnorm * average_jdotb;
jxbout.bdotb[jF] = dnorm1 * tjnorm * average_bdotb;

const double phipH_outside = vmec_internal_results.phipH[jHo];
const double phipH_inside = vmec_internal_results.phipH[jHi];

jxbout.bdotgradv[jF] =
dnorm1 * tjnorm * (phipH_outside + phipH_inside) / 2.0;

jxbout.jpar2[jF] = dnorm1 * tjnorm * average_jpar2;
jxbout.jperp2[jF] = dnorm1 * tjnorm * average_jperp2;

if (vmec_status == VmecStatus::SUCCESSFUL_TERMINATION ||
return_outputs_even_if_not_converged) {

jxbout.phin[jF] = vmec_internal_results.phiF[jF] /
vmec_internal_results.phiF[fc.ns - 1];

for (int kl = 0; kl < s.nZnT; ++kl) {
const int target_index = jF * s.nZnT + kl;

jxbout.jsupu3(target_index) = ovp * jxbout.itheta(target_index);
jxbout.jsupv3(target_index) = ovp * jxbout.izeta(target_index);

jxbout.bsupu3(target_index) = bsupu1[kl];
jxbout.bsupv3(target_index) = bsupv1[kl];

jxbout.jcrossb(target_index) = jxb[kl];
jxbout.jxb_gradp(target_index) = jxb[kl] - pprime_ovp;
jxbout.jdotb_sqrtg(target_index) = ovp * jxbout.bdotk(target_index);

jxbout.sqrtg3(target_index) = sqrtg[kl] * ovp;

jxbout.bsubs3(target_index) = bsubs_full.bsubs_full(target_index);
}
}
}

if (vmec_status == VmecStatus::SUCCESSFUL_TERMINATION ||
return_outputs_even_if_not_converged) {

for (int jH = 0; jH < vmec_internal_results.num_half - 1; ++jH) {

const double ovp = 2.0 /
(vmec_internal_results.dVdsH[jH + 1] +
vmec_internal_results.dVdsH[jH]) /
dnorm1;

for (int kl = 0; kl < s.nZnT; ++kl) {
const int target_index = jH * s.nZnT + kl;

const double bsubuv = covariant_b_derivatives.bsubuv(target_index);
const double bsubvu = covariant_b_derivatives.bsubvu(target_index);

jxbout.jsups3(target_index) = ovp * (bsubuv - bsubvu) / MU_0;

jxbout.bsubu3(target_index) = vmec_internal_results.bsubu(target_index);
jxbout.bsubv3(target_index) = vmec_internal_results.bsubv(target_index);
}
}
}

for (int kl = 0; kl < s.nZnT; ++kl) {

const int index_0 = 0 * s.nZnT + kl;
const int index_1 = 1 * s.nZnT + kl;
const int index_2 = 2 * s.nZnT + kl;

const int index_ns_1 = (fc.ns - 1) * s.nZnT + kl;
const int index_ns_2 = (fc.ns - 2) * s.nZnT + kl;
const int index_ns_3 = (fc.ns - 3) * s.nZnT + kl;

jxbout.izeta(index_0) = 2.0 * jxbout.izeta(index_1) - jxbout.izeta(index_2);
jxbout.izeta(index_ns_1) =
2.0 * jxbout.izeta(index_ns_2) - jxbout.izeta(index_ns_3);
}

jxbout.jdotb[0] = 2.0 * jxbout.jdotb[1] - jxbout.jdotb[2];
jxbout.jdotb[fc.ns - 1] =
2.0 * jxbout.jdotb[fc.ns - 2] - jxbout.jdotb[fc.ns - 3];

jxbout.bdotb[0] = 2.0 * jxbout.bdotb[1] - jxbout.bdotb[2];
jxbout.bdotb[fc.ns - 1] =
2.0 * jxbout.bdotb[fc.ns - 2] - jxbout.bdotb[fc.ns - 3];

jxbout.bdotgradv[0] = 2.0 * jxbout.bdotgradv[1] - jxbout.bdotgradv[2];
jxbout.bdotgradv[fc.ns - 1] =
2.0 * jxbout.bdotgradv[fc.ns - 2] - jxbout.bdotgradv[fc.ns - 3];

jxbout.pprim[0] = 2.0 * jxbout.pprim[1] - jxbout.pprim[2];
jxbout.pprim[fc.ns - 1] =
2.0 * jxbout.pprim[fc.ns - 2] - jxbout.pprim[fc.ns - 3];

return jxbout;
}

vmecpp::MercierStabilityIntermediateQuantities
vmecpp::ComputeIntermediateMercierQuantities(
const Sizes& s, const FlowControl& fc,
const VmecInternalResults& vmec_internal_results,
const JxBOutFileContents& jxbout) {

MercierStabilityIntermediateQuantities mercier_intermediate;

mercier_intermediate.s = VectorXd::Zero(fc.ns);
mercier_intermediate.shear = VectorXd::Zero(fc.ns);
mercier_intermediate.vpp = VectorXd::Zero(fc.ns);
mercier_intermediate.d_pressure_d_s = VectorXd::Zero(fc.ns);
mercier_intermediate.d_toroidal_current_d_s = VectorXd::Zero(fc.ns);
mercier_intermediate.phip_realH = VectorXd::Zero(fc.ns - 1);
mercier_intermediate.phip_realF = VectorXd::Zero(fc.ns);
mercier_intermediate.vp_real = VectorXd::Zero(fc.ns - 1);
mercier_intermediate.torcur = VectorXd::Zero(fc.ns - 1);

mercier_intermediate.gsqrt_full = RowMatrixXd::Zero(fc.ns, s.nZnT);
mercier_intermediate.bdotj = RowMatrixXd::Zero(fc.ns, s.nZnT);
mercier_intermediate.gpp = RowMatrixXd::Zero(fc.ns, s.nZnT);
mercier_intermediate.b2 = RowMatrixXd::Zero(fc.ns - 1, s.nZnT);

mercier_intermediate.tpp = VectorXd::Zero(fc.ns);
mercier_intermediate.tbb = VectorXd::Zero(fc.ns);
mercier_intermediate.tjb = VectorXd::Zero(fc.ns);
mercier_intermediate.tjj = VectorXd::Zero(fc.ns);

for (int jH = 0; jH < fc.ns - 1; ++jH) {

mercier_intermediate.phip_realH[jH] =
2.0 * M_PI * vmec_internal_results.phipH[jH] *
vmec_internal_results.sign_of_jacobian;

mercier_intermediate.vp_real[jH] =
vmec_internal_results.sign_of_jacobian * (4.0 * M_PI * M_PI) *
vmec_internal_results.dVdsH[jH] / mercier_intermediate.phip_realH[jH];

for (int kl = 0; kl < s.nZnT; ++kl) {
const int idx_kl = jH * s.nZnT + kl;
const int l = kl % s.nThetaEff;
mercier_intermediate.torcur[jH] +=
vmec_internal_results.bsubu(idx_kl) * s.wInt[l];
}
mercier_intermediate.torcur[jH] *=
vmec_internal_results.sign_of_jacobian * 2.0 * M_PI;
}

for (int jF = 1; jF < fc.ns - 1; ++jF) {
const int jHi = jF - 1;
const int jHo = jF;

mercier_intermediate.phip_realF[jF] =
(mercier_intermediate.phip_realH[jHo] +
mercier_intermediate.phip_realH[jHi]) /
2.0;
const double denom = mercier_intermediate.phip_realF[jF] * fc.deltaS;

mercier_intermediate.shear[jF] =
(vmec_internal_results.iotaH[jHo] - vmec_internal_results.iotaH[jHi]) /
denom;

mercier_intermediate.vpp[jF] = (mercier_intermediate.vp_real[jHo] -
mercier_intermediate.vp_real[jHi]) /
denom;

mercier_intermediate.d_pressure_d_s[jF] =
(vmec_internal_results.presH[jHo] - vmec_internal_results.presH[jHi]) /
denom;

mercier_intermediate.d_toroidal_current_d_s[jF] =
(mercier_intermediate.torcur[jHo] - mercier_intermediate.torcur[jHi]) /
denom;

for (int kl = 0; kl < s.nZnT; ++kl) {
const int index_full = jF * s.nZnT + kl;

const double gsqrt_outside =
vmec_internal_results.gsqrt(jHo * s.nZnT + kl);
const double gsqrt_inside =
vmec_internal_results.gsqrt(jHi * s.nZnT + kl);
mercier_intermediate.gsqrt_full(index_full) =
(gsqrt_outside + gsqrt_inside) / 2.0;

mercier_intermediate.bdotj(index_full) =
jxbout.bdotk(index_full) * MU_0 /
mercier_intermediate.gsqrt_full(index_full);

mercier_intermediate.gsqrt_full(index_full) /=
mercier_intermediate.phip_realF[jF];
}

mercier_intermediate.s[jF] = jF * fc.deltaS;
const double sqrtSF = std::sqrt(mercier_intermediate.s[jF]);

for (int kl = 0; kl < s.nZnT; ++kl) {
const int index_full = jF * s.nZnT + kl;

const double rtf = vmec_internal_results.ru_e(index_full) +
sqrtSF * vmec_internal_results.ru_o(index_full);

const double ztf = vmec_internal_results.zu_e(index_full) +
sqrtSF * vmec_internal_results.zu_o(index_full);

const double rzf = vmec_internal_results.rv_e(index_full) +
sqrtSF * vmec_internal_results.rv_o(index_full);

const double zzf = vmec_internal_results.zv_e(index_full) +
sqrtSF * vmec_internal_results.zv_o(index_full);

const double r1f = vmec_internal_results.r_e(index_full) +
sqrtSF * vmec_internal_results.r_o(index_full);

const double gtt = rtf * rtf + ztf * ztf;

const double gpp_numerator = mercier_intermediate.gsqrt_full(index_full) *
mercier_intermediate.gsqrt_full(index_full);

const double gpp_denominator_ingredient = rtf * zzf - rzf * ztf;
const double gpp_denominator =
gtt * r1f * r1f +
gpp_denominator_ingredient * gpp_denominator_ingredient;
mercier_intermediate.gpp(index_full) = gpp_numerator / gpp_denominator;
}
}

for (int jH = 0; jH < fc.ns - 1; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
const int index_half = jH * s.nZnT + kl;

mercier_intermediate.b2(index_half) =
2.0 * (vmec_internal_results.total_pressure(index_half) -
vmec_internal_results.presH[jH]);
}
}

const double four_pi_squared = 4.0 * M_PI * M_PI;
for (int jF = 1; jF < fc.ns - 1; ++jF) {
const int jHi = jF - 1;
const int jHo = jF;

for (int kl = 0; kl < s.nZnT; ++kl) {
const int l = kl % s.nThetaEff;
const int index_full = jF * s.nZnT + kl;
const int index_half_o = jHo * s.nZnT + kl;
const int index_half_i = jHi * s.nZnT + kl;

const double b2i = (mercier_intermediate.b2(index_half_o) +
mercier_intermediate.b2(index_half_i)) /
2.0;

const double ob2 = mercier_intermediate.gsqrt_full(index_full) / b2i;
mercier_intermediate.tpp[jF] += ob2 * s.wInt[l];

const double ob2_reused = b2i *
mercier_intermediate.gsqrt_full(index_full) *
mercier_intermediate.gpp(index_full);
mercier_intermediate.tbb[jF] += ob2_reused * s.wInt[l];

const double jdotb = mercier_intermediate.bdotj(index_full) *
mercier_intermediate.gpp(index_full) *
mercier_intermediate.gsqrt_full(index_full);
mercier_intermediate.tjb[jF] += jdotb * s.wInt[l];

const double jdotb_reused =
jdotb * mercier_intermediate.bdotj(index_full) / b2i;
mercier_intermediate.tjj[jF] += jdotb_reused * s.wInt[l];
}

mercier_intermediate.tpp[jF] *= four_pi_squared;
mercier_intermediate.tbb[jF] *= four_pi_squared;
mercier_intermediate.tjb[jF] *= four_pi_squared;
mercier_intermediate.tjj[jF] *= four_pi_squared;
}

return mercier_intermediate;
}

vmecpp::MercierFileContents vmecpp::ComputeMercierStability(
const FlowControl& fc, const VmecInternalResults& vmec_internal_results,
const MercierStabilityIntermediateQuantities& mercier_intermediate) {
MercierFileContents mercier;

mercier.s = VectorXd::Zero(fc.ns);
mercier.toroidal_flux = VectorXd::Zero(fc.ns);
mercier.iota = VectorXd::Zero(fc.ns);
mercier.shear = VectorXd::Zero(fc.ns);
mercier.d_volume_d_s = VectorXd::Zero(fc.ns);
mercier.well = VectorXd::Zero(fc.ns);
mercier.toroidal_current = VectorXd::Zero(fc.ns);
mercier.d_toroidal_current_d_s = VectorXd::Zero(fc.ns);
mercier.pressure = VectorXd::Zero(fc.ns);
mercier.d_pressure_d_s = VectorXd::Zero(fc.ns);

mercier.DMerc = VectorXd::Zero(fc.ns);
mercier.Dshear = VectorXd::Zero(fc.ns);
mercier.Dwell = VectorXd::Zero(fc.ns);
mercier.Dcurr = VectorXd::Zero(fc.ns);
mercier.Dgeod = VectorXd::Zero(fc.ns);

for (int jF = 1; jF < fc.ns - 1; ++jF) {
const int jHi = jF - 1;
const int jHo = jF;

mercier.s[jF] = mercier_intermediate.s[jF];

const double vp_full = (mercier_intermediate.vp_real[jHo] +
mercier_intermediate.vp_real[jHi]) /
2.0;
if (vp_full == 0.0) {

continue;
}

mercier.toroidal_flux[jF] = 0.0;
for (int jF_prime = 1; jF_prime <= jF; ++jF_prime) {
mercier.toroidal_flux[jF] += mercier_intermediate.phip_realF[jF_prime];
}
mercier.toroidal_flux[jF] *= fc.deltaS;

mercier.iota[jF] =
(vmec_internal_results.iotaH[jHo] + vmec_internal_results.iotaH[jHi]) /
2.0;

mercier.shear[jF] = mercier_intermediate.shear[jF] / vp_full;

mercier.d_volume_d_s[jF] = vp_full;

mercier.well[jF] =
-mercier_intermediate.vpp[jF] * vmec_internal_results.sign_of_jacobian;

mercier.toroidal_current[jF] =
(mercier_intermediate.torcur[jHo] + mercier_intermediate.torcur[jHi]) /
2.0;

mercier.d_toroidal_current_d_s[jF] =
mercier_intermediate.d_toroidal_current_d_s[jF] / vp_full;

mercier.pressure[jF] =
(vmec_internal_results.presH[jHo] + vmec_internal_results.presH[jHi]) /
2.0;

mercier.d_pressure_d_s[jF] =
mercier_intermediate.d_pressure_d_s[jF] / vp_full;
}

for (int jF = 1; jF < fc.ns - 1; ++jF) {

const double tpp = mercier_intermediate.tpp[jF];
const double tjb = mercier_intermediate.tjb[jF];
const double tbb = mercier_intermediate.tbb[jF];
const double tjj = mercier_intermediate.tjj[jF];

mercier.Dshear[jF] =
mercier_intermediate.shear[jF] * mercier_intermediate.shear[jF] / 4.0;

mercier.Dcurr[jF] =
-mercier_intermediate.shear[jF] *
(tjb - mercier_intermediate.d_toroidal_current_d_s[jF] * tbb);

mercier.Dwell[jF] = mercier_intermediate.d_pressure_d_s[jF] *
(mercier_intermediate.vpp[jF] -
mercier_intermediate.d_pressure_d_s[jF] * tpp) *
tbb;

mercier.Dgeod[jF] = tjb * tjb - tbb * tjj;

mercier.DMerc[jF] = mercier.Dshear[jF] + mercier.Dcurr[jF] +
mercier.Dwell[jF] + mercier.Dgeod[jF];
}

mercier.s[0] = 0.0;
mercier.s[fc.ns - 1] = 1.0;

return mercier;
}

vmecpp::Threed1FirstTableIntermediate
vmecpp::ComputeIntermediateThreed1FirstTableQuantities(
const Sizes& s, const FlowControl& fc,
const VmecInternalResults& vmec_internal_results) {
Threed1FirstTableIntermediate threed1_first_table_intermediate;

threed1_first_table_intermediate.tau = RowMatrixXd::Zero(fc.ns - 1, s.nZnT);

threed1_first_table_intermediate.beta_vol = VectorXd::Zero(fc.ns - 1);
threed1_first_table_intermediate.overr = VectorXd::Zero(fc.ns - 1);
threed1_first_table_intermediate.bvcoH = VectorXd::Zero(fc.ns - 1);
threed1_first_table_intermediate.bucoH = VectorXd::Zero(fc.ns - 1);

threed1_first_table_intermediate.presf = VectorXd::Zero(fc.ns);
threed1_first_table_intermediate.phipf_loc = VectorXd::Zero(fc.ns);
threed1_first_table_intermediate.phi1 = VectorXd::Zero(fc.ns);
threed1_first_table_intermediate.chi1 = VectorXd::Zero(fc.ns);
threed1_first_table_intermediate.chi = VectorXd::Zero(fc.ns);
threed1_first_table_intermediate.jcurv = VectorXd::Zero(fc.ns);
threed1_first_table_intermediate.jcuru = VectorXd::Zero(fc.ns);
threed1_first_table_intermediate.presgrad = VectorXd::Zero(fc.ns);
threed1_first_table_intermediate.vpphi = VectorXd::Zero(fc.ns);
threed1_first_table_intermediate.equif = VectorXd::Zero(fc.ns);
threed1_first_table_intermediate.bucof = VectorXd::Zero(fc.ns);
threed1_first_table_intermediate.bvcof = VectorXd::Zero(fc.ns);

for (int jH = 0; jH < fc.ns - 1; ++jH) {
double s2 = 0.0;
double avg_tau_over_r12 = 0.0;
for (int kl = 0; kl < s.nZnT; ++kl) {
const int l = kl % s.nThetaEff;
const int index_half = jH * s.nZnT + kl;

threed1_first_table_intermediate.tau(index_half) =
vmec_internal_results.sign_of_jacobian * s.wInt[l] *
vmec_internal_results.gsqrt(index_half);

s2 += vmec_internal_results.total_pressure(index_half) *
threed1_first_table_intermediate.tau(index_half);

avg_tau_over_r12 += threed1_first_table_intermediate.tau(index_half) /
vmec_internal_results.r12(index_half);
}
s2 /= vmec_internal_results.dVdsH[jH];
s2 -= vmec_internal_results.presH[jH];

threed1_first_table_intermediate.beta_vol[jH] =
vmec_internal_results.presH[jH] / s2;

threed1_first_table_intermediate.overr[jH] =
avg_tau_over_r12 / vmec_internal_results.dVdsH[jH];
}

threed1_first_table_intermediate.beta_axis =
1.5 * threed1_first_table_intermediate.beta_vol[0] -
0.5 * threed1_first_table_intermediate.beta_vol[1];

threed1_first_table_intermediate.presf[0] =
1.5 * vmec_internal_results.presH[0] -
0.5 * vmec_internal_results.presH[1];
threed1_first_table_intermediate.phipf_loc[0] =
2.0 * M_PI * vmec_internal_results.sign_of_jacobian *
(1.5 * vmec_internal_results.phipH[0] -
0.5 * vmec_internal_results.phipH[1]);
for (int jF = 1; jF < fc.ns - 1; ++jF) {
const int jHi = jF - 1;
const int jHo = jF;

threed1_first_table_intermediate.presf[jF] =
(vmec_internal_results.presH[jHo] + vmec_internal_results.presH[jHi]) /
2.0;
threed1_first_table_intermediate.phipf_loc[jF] =
2.0 * M_PI * vmec_internal_results.sign_of_jacobian *
(vmec_internal_results.phipH[jHo] + vmec_internal_results.phipH[jHi]) /
2.0;
}
threed1_first_table_intermediate.presf[fc.ns - 1] =
1.5 * vmec_internal_results.presH[fc.ns - 2] -
0.5 * vmec_internal_results.presH[fc.ns - 3];
threed1_first_table_intermediate.phipf_loc[fc.ns - 1] =
2.0 * M_PI * vmec_internal_results.sign_of_jacobian *
(1.5 * vmec_internal_results.phipH[fc.ns - 2] -
0.5 * vmec_internal_results.phipH[fc.ns - 3]);

threed1_first_table_intermediate.phi1[0] = 0.0;
threed1_first_table_intermediate.chi1[0] = 0.0;
for (int jF = 1; jF < fc.ns; ++jF) {
const int jHi = jF - 1;

threed1_first_table_intermediate.phi1[jF] =
threed1_first_table_intermediate.phi1[jF - 1] +
vmec_internal_results.phipH[jHi] * fc.deltaS;
threed1_first_table_intermediate.chi1[jF] =
threed1_first_table_intermediate.chi1[jF - 1] +
(vmec_internal_results.phipH[jHi] * vmec_internal_results.iotaH[jHi]) *
fc.deltaS;

threed1_first_table_intermediate.chi[jF] =
2.0 * M_PI * threed1_first_table_intermediate.chi1[jF];
}

threed1_first_table_intermediate.bucoH.resize(fc.ns - 1);
threed1_first_table_intermediate.bvcoH.resize(fc.ns - 1);
for (int jH = 0; jH < fc.ns - 1; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
int iHalf = jH * s.nZnT + kl;
int l = kl % s.nThetaEff;
threed1_first_table_intermediate.bucoH[jH] +=
vmec_internal_results.bsubu(iHalf) * s.wInt[l];
threed1_first_table_intermediate.bvcoH[jH] +=
vmec_internal_results.bsubv(iHalf) * s.wInt[l];
}
}

double signByDeltaS = vmec_internal_results.sign_of_jacobian / fc.deltaS;

for (int jF = 1; jF < fc.ns - 1; ++jF) {
const int jHi = jF - 1;
const int jHo = jF;

threed1_first_table_intermediate.jcurv[jF] =
signByDeltaS * (threed1_first_table_intermediate.bucoH[jHo] -
threed1_first_table_intermediate.bucoH[jHi]);
threed1_first_table_intermediate.jcuru[jF] =
-signByDeltaS * (threed1_first_table_intermediate.bvcoH[jHo] -
threed1_first_table_intermediate.bvcoH[jHi]);

threed1_first_table_intermediate.presgrad[jF] =
(vmec_internal_results.presH[jHo] - vmec_internal_results.presH[jHi]) /
fc.deltaS;

threed1_first_table_intermediate.vpphi[jF] =
0.5 *
(vmec_internal_results.dVdsH[jHo] + vmec_internal_results.dVdsH[jHi]);

threed1_first_table_intermediate.equif[jF] =
(vmec_internal_results.chipF[jF] *
threed1_first_table_intermediate.jcurv[jF] -
vmec_internal_results.phipF[jF] *
threed1_first_table_intermediate.jcuru[jF]) /
threed1_first_table_intermediate.vpphi[jF] +
threed1_first_table_intermediate.presgrad[jF];
}

threed1_first_table_intermediate.bucof[0] = 0.0;
threed1_first_table_intermediate.bvcof[0] =
1.5 * threed1_first_table_intermediate.bvcoH[0] -
0.5 * threed1_first_table_intermediate.bvcoH[1];
for (int jF = 1; jF < fc.ns - 1; ++jF) {
const int jHi = jF - 1;
const int jHo = jF;

threed1_first_table_intermediate.equif[jF] *=
threed1_first_table_intermediate.vpphi[jF] /
(std::abs(threed1_first_table_intermediate.jcurv[jF] *
vmec_internal_results.chipF[jF]) +
std::abs(threed1_first_table_intermediate.jcuru[jF] *
vmec_internal_results.phipF[jF]) +
std::abs(threed1_first_table_intermediate.presgrad[jF] *
threed1_first_table_intermediate.vpphi[jF]));

threed1_first_table_intermediate.bucof[jF] =
(threed1_first_table_intermediate.bucoH[jHo] +
threed1_first_table_intermediate.bucoH[jHi]) /
2.0;
threed1_first_table_intermediate.bvcof[jF] =
(threed1_first_table_intermediate.bvcoH[jHo] +
threed1_first_table_intermediate.bvcoH[jHi]) /
2.0;
}
threed1_first_table_intermediate.bucof[fc.ns - 1] =
1.5 * threed1_first_table_intermediate.bucoH[fc.ns - 2] -
0.5 * threed1_first_table_intermediate.bucoH[fc.ns - 3];
threed1_first_table_intermediate.bvcof[fc.ns - 1] =
1.5 * threed1_first_table_intermediate.bvcoH[fc.ns - 2] -
0.5 * threed1_first_table_intermediate.bvcoH[fc.ns - 3];

threed1_first_table_intermediate.equif[0] =
2.0 * threed1_first_table_intermediate.equif[1] -
threed1_first_table_intermediate.equif[2];
threed1_first_table_intermediate.equif[fc.ns - 1] =
2.0 * threed1_first_table_intermediate.equif[fc.ns - 2] -
threed1_first_table_intermediate.equif[fc.ns - 3];

threed1_first_table_intermediate.jcurv[0] =
2.0 * threed1_first_table_intermediate.jcurv[1] -
threed1_first_table_intermediate.jcurv[2];
threed1_first_table_intermediate.jcurv[fc.ns - 1] =
2.0 * threed1_first_table_intermediate.jcurv[fc.ns - 2] -
threed1_first_table_intermediate.jcurv[fc.ns - 3];

threed1_first_table_intermediate.jcuru[0] =
2.0 * threed1_first_table_intermediate.jcuru[1] -
threed1_first_table_intermediate.jcuru[2];
threed1_first_table_intermediate.jcuru[fc.ns - 1] =
2.0 * threed1_first_table_intermediate.jcuru[fc.ns - 2] -
threed1_first_table_intermediate.jcuru[fc.ns - 3];

threed1_first_table_intermediate.presgrad[0] =
2.0 * threed1_first_table_intermediate.presgrad[1] -
threed1_first_table_intermediate.presgrad[2];
threed1_first_table_intermediate.presgrad[fc.ns - 1] =
2.0 * threed1_first_table_intermediate.presgrad[fc.ns - 2] -
threed1_first_table_intermediate.presgrad[fc.ns - 3];

threed1_first_table_intermediate.vpphi[0] =
2.0 * threed1_first_table_intermediate.vpphi[1] -
threed1_first_table_intermediate.vpphi[2];
threed1_first_table_intermediate.vpphi[fc.ns - 1] =
2.0 * threed1_first_table_intermediate.vpphi[fc.ns - 2] -
threed1_first_table_intermediate.vpphi[fc.ns - 3];

return threed1_first_table_intermediate;
}

vmecpp::Threed1FirstTable vmecpp::ComputeThreed1FirstTable(
const FlowControl& fc, const VmecInternalResults& vmec_internal_results,
const JxBOutFileContents& jxbout,
const Threed1FirstTableIntermediate& threed1_first_table_intermediate) {
const double fac = 2.0 * M_PI * vmec_internal_results.sign_of_jacobian;

Threed1FirstTable threed1_first_table;

threed1_first_table.s = VectorXd::Zero(fc.ns);
threed1_first_table.radial_force = VectorXd::Zero(fc.ns);
threed1_first_table.toroidal_flux = VectorXd::Zero(fc.ns);
threed1_first_table.iota = VectorXd::Zero(fc.ns);
threed1_first_table.avg_jsupu = VectorXd::Zero(fc.ns);
threed1_first_table.avg_jsupv = VectorXd::Zero(fc.ns);
threed1_first_table.d_volume_d_phi = VectorXd::Zero(fc.ns);
threed1_first_table.d_pressure_d_phi = VectorXd::Zero(fc.ns);
threed1_first_table.spectral_width = VectorXd::Zero(fc.ns);
threed1_first_table.pressure = VectorXd::Zero(fc.ns);
threed1_first_table.buco_full = VectorXd::Zero(fc.ns);
threed1_first_table.bvco_full = VectorXd::Zero(fc.ns);
threed1_first_table.j_dot_b = VectorXd::Zero(fc.ns);
threed1_first_table.b_dot_b = VectorXd::Zero(fc.ns);

for (int jF = 0; jF < fc.ns; ++jF) {

const double s = jF * fc.deltaS;

const double vpphi = threed1_first_table_intermediate.vpphi[jF];
const double phipf_loc = threed1_first_table_intermediate.phipf_loc[jF];

const double cur0 = fac * vpphi * 2.0 * M_PI;

threed1_first_table.s[jF] = s;

threed1_first_table.radial_force[jF] =
threed1_first_table_intermediate.equif[jF];

threed1_first_table.toroidal_flux[jF] =
fac * threed1_first_table_intermediate.phi1[jF];

threed1_first_table.iota[jF] = vmec_internal_results.iotaF[jF];

threed1_first_table.avg_jsupu[jF] =
threed1_first_table_intermediate.jcuru[jF] / vpphi / MU_0;

threed1_first_table.avg_jsupv[jF] =
threed1_first_table_intermediate.jcurv[jF] / vpphi / MU_0;

threed1_first_table.d_volume_d_phi[jF] = cur0 / phipf_loc;

threed1_first_table.d_pressure_d_phi[jF] =
threed1_first_table_intermediate.presgrad[jF] / phipf_loc / MU_0;

threed1_first_table.spectral_width[jF] =
vmec_internal_results.spectral_width[jF];

threed1_first_table.pressure[jF] =
threed1_first_table_intermediate.presf[jF] / MU_0;

threed1_first_table.buco_full[jF] =
threed1_first_table_intermediate.bucof[jF];

threed1_first_table.bvco_full[jF] =
threed1_first_table_intermediate.bvcof[jF];

threed1_first_table.j_dot_b[jF] = jxbout.jdotb[jF];

threed1_first_table.b_dot_b[jF] = jxbout.bdotb[jF];
}

return threed1_first_table;
}

vmecpp::Threed1GeometricAndMagneticQuantitiesIntermediate
vmecpp::ComputeIntermediateThreed1GeometricMagneticQuantities(
const Sizes& s, const FlowControl& fc,
const HandoverStorage& handover_storage,
const VmecInternalResults& vmec_internal_results,
const JxBOutFileContents& jxbout,
const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
VacuumPressureState vacuum_pressure_state) {
Threed1GeometricAndMagneticQuantitiesIntermediate intermediate;

intermediate.anorm = 2.0 * M_PI * fc.deltaS;
intermediate.vnorm = 2.0 * M_PI * intermediate.anorm;

intermediate.surf_area = VectorXd::Zero(s.nZnT);
intermediate.circumference_sum = 0.0;
for (int kl = 0; kl < s.nZnT; ++kl) {
const int l = kl % s.nThetaEff;
const int lcfs_kl = (fc.ns - 1) * s.nZnT + kl;

const double ru0 = vmec_internal_results.ruFull(lcfs_kl);
const double zu0 = vmec_internal_results.zuFull(lcfs_kl);
const double guu_1u = ru0 * ru0 + zu0 * zu0;

intermediate.circumference_sum += std::sqrt(guu_1u) * s.wInt[l];

const double r =
vmec_internal_results.r_e(lcfs_kl) + vmec_internal_results.r_o(lcfs_kl);
const double rv = vmec_internal_results.rv_e(lcfs_kl) +
vmec_internal_results.rv_o(lcfs_kl);
const double zv = vmec_internal_results.zv_e(lcfs_kl) +
vmec_internal_results.zv_o(lcfs_kl);

const double rv_zu_minus_zv_ru = rv * zu0 - zv * ru0;

intermediate.surf_area[kl] =
s.wInt[l] *
std::sqrt(r * r * guu_1u + rv_zu_minus_zv_ru * rv_zu_minus_zv_ru);
}

double sump_sum = 0.0;
for (int jH = 0; jH < fc.ns - 1; ++jH) {
sump_sum +=
vmec_internal_results.dVdsH[jH] * vmec_internal_results.presH[jH];
}
intermediate.sump = intermediate.vnorm * sump_sum;

intermediate.btor_vac = VectorXd::Zero(s.nZnT);
intermediate.btor1 = VectorXd::Zero(s.nZnT);
intermediate.dbtor = VectorXd::Zero(s.nZnT);
intermediate.phat = VectorXd::Zero(s.nZnT);

intermediate.delphid_exact = 0.0;
intermediate.musubi = 0.0;
intermediate.rshaf1 = 0.0;
intermediate.rshaf2 = 0.0;
for (int jH = 0; jH < fc.ns - 1; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
const int index_half = jH * s.nZnT + kl;

intermediate.btor_vac[kl] =
handover_storage.rBtor / vmec_internal_results.r12(index_half);

intermediate.delphid_exact +=
(intermediate.btor_vac[kl] / vmec_internal_results.r12(index_half) -
vmec_internal_results.bsupv(index_half)) *
threed1_first_table_intermediate.tau(index_half);

intermediate.btor1[kl] = vmec_internal_results.r12(index_half) *
vmec_internal_results.bsupv(index_half);

intermediate.dbtor[kl] =
intermediate.btor1[kl] * intermediate.btor1[kl] -
intermediate.btor_vac[kl] * intermediate.btor_vac[kl];

intermediate.musubi -= intermediate.dbtor[kl] *
threed1_first_table_intermediate.tau(index_half);

intermediate.phat[kl] =
vmec_internal_results.total_pressure(index_half) -
0.5 * intermediate.btor_vac[kl] * intermediate.btor_vac[kl];
intermediate.phat[kl] -= intermediate.dbtor[kl];
intermediate.phat[kl] *= threed1_first_table_intermediate.tau(index_half);

intermediate.rshaf1 += intermediate.phat[kl];
intermediate.rshaf2 +=
intermediate.phat[kl] / vmec_internal_results.r12(index_half);
}
}
intermediate.delphid_exact *= intermediate.anorm;
intermediate.rshaf = intermediate.rshaf1 / intermediate.rshaf2;

intermediate.fpsi0 = threed1_first_table_intermediate.bvcof[0];

intermediate.redge = VectorXd::Zero(s.nZnT);
for (int kl = 0; kl < s.nZnT; ++kl) {
const int lcfs_kl = (fc.ns - 1) * s.nZnT + kl;
intermediate.redge[kl] =
vmec_internal_results.r_e(lcfs_kl) + vmec_internal_results.r_o(lcfs_kl);
}
if (fc.lfreeb && vacuum_pressure_state >= VacuumPressureState::kActive) {
for (int k = 0; k < s.nZeta; ++k) {
for (int l = 0; l < s.nThetaEff; ++l) {

int idx_kl = k * s.nThetaEff + l;
int idx_lk = l * s.nZeta + k;

const double bsubvvac_r =
handover_storage.bSubVVac / intermediate.redge[idx_kl];
intermediate.phat[idx_kl] =
handover_storage.vacuum_magnetic_pressure[idx_lk] -
0.5 * bsubvvac_r * bsubvvac_r;
}
}
} else {
for (int kl = 0; kl < s.nZnT; ++kl) {
const int last_half = (fc.ns - 2) * s.nZnT + kl;
const int last1_half = (fc.ns - 3) * s.nZnT + kl;
const double bsq_at_lcfs =
1.5 * vmec_internal_results.total_pressure(last_half) -
0.5 * vmec_internal_results.total_pressure(last1_half);
const double btor_edge = handover_storage.rBtor / intermediate.redge[kl];
intermediate.phat[kl] = bsq_at_lcfs - 0.5 * btor_edge * btor_edge;
}
}

double bsq_tau_sum = 0.0;
double btor_sum = 0.0;
double p2_sum = 0.0;
for (int jH = 0; jH < fc.ns - 1; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
const int index_half = jH * s.nZnT + kl;

const double tau = threed1_first_table_intermediate.tau(index_half);
bsq_tau_sum += vmec_internal_results.total_pressure(index_half) * tau;

const double r_bsupv = vmec_internal_results.r12(index_half) *
vmec_internal_results.bsupv(index_half);
btor_sum += tau * r_bsupv * r_bsupv;
}

const double pressure = vmec_internal_results.presH[jH];
p2_sum += pressure * pressure * vmec_internal_results.dVdsH[jH];
}

intermediate.sumbtot =
2.0 * (intermediate.vnorm * bsq_tau_sum - intermediate.sump);
intermediate.sumbtor = intermediate.vnorm * btor_sum;
intermediate.sumbpol = intermediate.sumbtot - intermediate.sumbtor;

intermediate.sump20 = 2.0 * intermediate.sump;
intermediate.sump2 = intermediate.vnorm * p2_sum;

intermediate.jPS2.resize(fc.ns);

intermediate.jPS2.setZero();
intermediate.jpar_perp_sum = 0.0;
intermediate.jparPS_perp_sum = 0.0;
intermediate.s2 = 0.0;
for (int jF = 1; jF < fc.ns - 1; ++jF) {
const int jHi = jF - 1;
const int jHo = jF;

intermediate.jPS2[jF] = jxbout.jpar2[jF] - jxbout.jdotb[jF] *
jxbout.jdotb[jF] /
jxbout.bdotb[jF];

const double two_dVds_full =
vmec_internal_results.dVdsH[jHo] + vmec_internal_results.dVdsH[jHi];
intermediate.jpar_perp_sum += jxbout.jpar2[jF] * two_dVds_full;
intermediate.jparPS_perp_sum += intermediate.jPS2[jF] * two_dVds_full;
intermediate.s2 += jxbout.jperp2[jF] * two_dVds_full;
}

intermediate.fac =
2.0 * M_PI * fc.deltaS * vmec_internal_results.sign_of_jacobian;
intermediate.r3v = VectorXd::Zero(fc.ns - 1);
for (int jH = 0; jH < fc.ns - 1; ++jH) {
intermediate.r3v[jH] = intermediate.fac * vmec_internal_results.phipH[jH] *
vmec_internal_results.iotaH[jH];
}

return intermediate;
}

vmecpp::Threed1GeometricAndMagneticQuantities
vmecpp::ComputeThreed1GeometricMagneticQuantities(
const Sizes& s, const FlowControl& fc,
const HandoverStorage& handover_storage,
const VmecInternalResults& vmec_internal_results,
const JxBOutFileContents& jxbout,
const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
const Threed1GeometricAndMagneticQuantitiesIntermediate& intermediate) {
Threed1GeometricAndMagneticQuantities result;

double toroidal_flux_sum = 0.0;
for (int jH = 0; jH < fc.ns - 1; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
const int index_half = jH * s.nZnT + kl;

toroidal_flux_sum += vmec_internal_results.bsupv(index_half) *
threed1_first_table_intermediate.tau(index_half);
}
}
result.toroidal_flux = intermediate.anorm * toroidal_flux_sum;

result.circum_p = 2.0 * M_PI * intermediate.circumference_sum;

double surface_area_sum = 0.0;
for (int kl = 0; kl < s.nZnT; ++kl) {
surface_area_sum += intermediate.surf_area[kl];
}
result.surf_area_p = 4.0 * M_PI * M_PI * surface_area_sum;

double cross_section_area_sum = 0.0;
double volume_sum = 0.0;
for (int kl = 0; kl < s.nZnT; ++kl) {
const int l = kl % s.nThetaEff;
const int lcfs_kl = (fc.ns - 1) * s.nZnT + kl;

const double rb =
vmec_internal_results.r_e(lcfs_kl) + vmec_internal_results.r_o(lcfs_kl);
const double zub = vmec_internal_results.zu_e(lcfs_kl) +
vmec_internal_results.zu_o(lcfs_kl);
const double t1 = rb * zub * s.wInt[l];

cross_section_area_sum += t1;
volume_sum += rb * t1;
}
result.cross_area_p = 2.0 * M_PI * std::abs(cross_section_area_sum);
result.volume_p = 2.0 * M_PI * M_PI * std::abs(volume_sum);

result.Rmajor_p = result.volume_p / (2.0 * M_PI * result.cross_area_p);
result.Aminor_p = std::sqrt(result.cross_area_p / M_PI);

result.aspect = result.Rmajor_p / result.Aminor_p;

const double d_of_kappa =
result.surf_area_p * result.Aminor_p / (2 * result.volume_p);
const double d_of_kappa_sqared = d_of_kappa * d_of_kappa;
result.kappa_p =
1.0 +
(M_PI * M_PI / 8.0) *
(d_of_kappa_sqared +
std::sqrt(std::abs(d_of_kappa_sqared * d_of_kappa_sqared - 1.0)) -
1.0);

RadialExtent radial_extent = handover_storage.GetRadialExtent();
result.rcen = (radial_extent.r_outer + radial_extent.r_inner) / 2.0;

const GeometricOffset& geometric_offset =
handover_storage.GetGeometricOffset();
result.aminr1 = std::sqrt(2.0 * result.volume_p /
(4.0 * M_PI * M_PI * geometric_offset.r_00));

result.pavg = intermediate.sump / result.volume_p;
result.factor = result.pavg * 2;

result.b0 = intermediate.fpsi0 / geometric_offset.r_00;

result.rmax_surf = 0.0;
result.rmin_surf = DBL_MAX;
result.zmax_surf = 0.0;
for (int kl = 0; kl < s.nZnT; ++kl) {
const int lcfs_kl = (fc.ns - 1) * s.nZnT + kl;
const double r =
vmec_internal_results.r_e(lcfs_kl) + vmec_internal_results.r_o(lcfs_kl);
const double z =
vmec_internal_results.z_e(lcfs_kl) + vmec_internal_results.z_o(lcfs_kl);
result.rmax_surf = std::max(result.rmax_surf, r);
result.rmin_surf = std::min(result.rmin_surf, r);
result.zmax_surf = std::max(result.zmax_surf, std::abs(z));
}

result.bmin = RowMatrixXd::Ones(fc.ns - 1, s.nThetaEff) * DBL_MAX;
result.bmax = RowMatrixXd::Zero(fc.ns - 1, s.nThetaEff);

for (int jH = 0; jH < fc.ns - 1; ++jH) {
for (int k = 0; k < s.nZeta; ++k) {
for (int l = 0; l < s.nThetaEff; ++l) {
const int kl = k * s.nThetaEff + l;
const int index_half = jH * s.nZnT + kl;

const double mod_b =
std::sqrt(2.0 * (vmec_internal_results.total_pressure(index_half) -
vmec_internal_results.presH[jH]));
result.bmax(jH * s.nThetaEff + l) =
std::max(result.bmax(jH * s.nThetaEff + l), mod_b);
result.bmin(jH * s.nThetaEff + l) =
std::min(result.bmin(jH * s.nThetaEff + l), mod_b);
}
}
}

int symmetry_planes_count = 1;
if (s.nZeta > 1) {
symmetry_planes_count = 2;
}
result.waist = VectorXd::Zero(symmetry_planes_count);
result.height = VectorXd::Zero(symmetry_planes_count);

int symmetry_plane_index = 0;
for (int k = 0; k < s.nZeta / 2 + 1; ++k) {
if (k != 0 && k != s.nZeta / 2) {
continue;
}

const int index_outboard = ((fc.ns - 1) * s.nZeta + k) * s.nThetaEff + 0;
const double r_outboard = vmec_internal_results.r_e(index_outboard) +
vmec_internal_results.r_o(index_outboard);

const int index_inboard =
((fc.ns - 1) * s.nZeta + k) * s.nThetaEff + (s.nThetaReduced - 1);
const double r_inboard = vmec_internal_results.r_e(index_inboard) +
vmec_internal_results.r_o(index_inboard);

result.waist[symmetry_plane_index] = r_outboard - r_inboard;

result.height[symmetry_plane_index] = 0.0;
for (int l = 0; l < s.nThetaEff; ++l) {
const int index_zeta = ((fc.ns - 1) * s.nZeta + k) * s.nThetaEff + l;
const double z = vmec_internal_results.z_e(index_zeta) +
vmec_internal_results.z_o(index_zeta);
result.height[symmetry_plane_index] =
std::max(result.height[symmetry_plane_index], std::abs(z));
}
result.height[symmetry_plane_index] *= 2.0;

symmetry_plane_index++;
}

result.betapol = 2.0 * intermediate.sump / intermediate.sumbpol;
result.betatot = intermediate.sump20 / intermediate.sumbtot;
result.betator = intermediate.sump20 / intermediate.sumbtor;
result.VolAvgB = std::sqrt(std::abs(intermediate.sumbtot / result.volume_p));

result.IonLarmor = 3.2e-3 / result.VolAvgB;

if (intermediate.s2 != 0.0) {
result.jpar_perp = intermediate.jpar_perp_sum / intermediate.s2;
result.jparPS_perp = intermediate.jparPS_perp_sum / intermediate.s2;
} else {
result.jpar_perp = intermediate.jpar_perp_sum;
result.jparPS_perp = intermediate.jparPS_perp_sum;
}

result.toroidal_current = handover_storage.cTor / MU_0;

result.rbtor = handover_storage.rBtor;
result.rbtor0 = handover_storage.rBtor0;

result.psi = VectorXd::Zero(fc.ns);
for (int jF = 1; jF < fc.ns; ++jF) {
const int jFi = jF - 1;
const int jHi = jF - 1;
result.psi[jF] = result.psi[jFi] + intermediate.r3v[jHi];
}

result.loc_jpar_perp = VectorXd::Zero(fc.ns);
result.loc_jparPS_perp = VectorXd::Zero(fc.ns);
for (int jF = 1; jF < fc.ns; ++jF) {
double jperp2 = DBL_EPSILON;
if (jxbout.jperp2[jF] != 0.0) {

jperp2 = jxbout.jperp2[jF];
}

result.loc_jpar_perp[jF] = jxbout.jpar2[jF] / jperp2;
if (jF < fc.ns - 1) {
result.loc_jparPS_perp[jF] = intermediate.jPS2[jF] / jperp2;
} else {
result.loc_jparPS_perp[jF] = 0.0;
}
}

result.ygeo = VectorXd::Zero(2 * fc.ns);
result.yinden = VectorXd::Zero(2 * fc.ns);
result.yellip = VectorXd::Zero(2 * fc.ns);
result.ytrian = VectorXd::Zero(2 * fc.ns);
result.yshift = VectorXd::Zero(2 * fc.ns);
for (int nplanes = 0; nplanes < 2; ++nplanes) {

int k = 0;
if (nplanes > 0) {
if (s.nZeta == 1) {
break;
}

k = s.nZeta / 2;
}

for (int jF = 1; jF < fc.ns; ++jF) {
double minimum_z = DBL_MAX;
double maximum_z = -DBL_MAX;
double minimum_r = DBL_MAX;
double maximum_r = -DBL_MAX;

double rzmax = 0.0;

const int num_passes = s.lasym ? 1 : 2;
const int num_theta = s.lasym ? s.nThetaEff : s.nThetaReduced;
for (int icount = 0; icount < num_passes; ++icount) {
int k1 = k;
int t1 = 1;
if (icount == 1) {

k1 = (s.nZeta - k) % s.nZeta;
t1 = -1;
}

for (int l = 0; l < num_theta; ++l) {
const int l_off = (jF * s.nZeta + k1) * s.nThetaEff + l;

const double yr1u = vmec_internal_results.r_e(l_off) +
vmec_internal_results.sqrtSF[jF] *
vmec_internal_results.r_o(l_off);
const double yz1u = t1 * (vmec_internal_results.z_e(l_off) +
vmec_internal_results.sqrtSF[jF] *
vmec_internal_results.z_o(l_off));

if (yz1u >= maximum_z) {
maximum_z = std::abs(yz1u);
rzmax = yr1u;
} else if (yz1u <= minimum_z) {
minimum_z = yz1u;

}

if (yr1u >= maximum_r) {
maximum_r = yr1u;

} else if (yr1u <= minimum_r) {
minimum_r = yr1u;

}
}
}

const int l_pi = (jF * s.nZeta + k) * s.nThetaEff + (s.nThetaReduced - 1);
double xmida =
vmec_internal_results.r_e(l_pi) +
vmec_internal_results.sqrtSF[jF] * vmec_internal_results.r_o(l_pi);

const int l_t = (jF * s.nZeta + k) * s.nThetaEff + 0;
double xmidb =
vmec_internal_results.r_e(l_t) +
vmec_internal_results.sqrtSF[jF] * vmec_internal_results.r_o(l_t);

const double rgeo = (xmidb + xmida) / 2.0;

result.ygeo[nplanes * fc.ns + jF] = (xmidb - xmida) / 2.0;

result.yinden[nplanes * fc.ns + jF] =
(xmida - minimum_r) / (maximum_r - minimum_r);

result.yellip[nplanes * fc.ns + jF] =
(maximum_z - minimum_z) / (maximum_r - minimum_r);

result.ytrian[nplanes * fc.ns + jF] =
(rgeo - rzmax) / (maximum_r - minimum_r);

const double r_axis = vmec_internal_results.r_e(k * s.nThetaEff + 0);
result.yshift[nplanes * fc.ns + jF] =
(r_axis - rgeo) / (maximum_r - minimum_r);
}
}

return result;
}

vmecpp::Threed1Volumetrics vmecpp::ComputeThreed1Volumetrics(
const Threed1GeometricAndMagneticQuantitiesIntermediate&
threed1_geometric_magnetic_intermediate,
const Threed1GeometricAndMagneticQuantities& threed1_geomag) {
const double fac = 0.5 / MU_0;

Threed1Volumetrics result;

result.int_p = threed1_geometric_magnetic_intermediate.sump / MU_0;
result.avg_p = threed1_geomag.pavg / MU_0;

result.int_bpol = fac * threed1_geometric_magnetic_intermediate.sumbpol;
result.avg_bpol = fac * threed1_geometric_magnetic_intermediate.sumbpol /
threed1_geomag.volume_p;

result.int_btor = fac * threed1_geometric_magnetic_intermediate.sumbtor;
result.avg_btor = fac * threed1_geometric_magnetic_intermediate.sumbtor /
threed1_geomag.volume_p;

result.int_modb = fac * threed1_geometric_magnetic_intermediate.sumbtot;
result.avg_modb = fac * threed1_geometric_magnetic_intermediate.sumbtot /
threed1_geomag.volume_p;

result.int_ekin = 1.5 * threed1_geometric_magnetic_intermediate.sump / MU_0;
result.avg_ekin = 1.5 * threed1_geomag.pavg / MU_0;

return result;
}

vmecpp::Threed1AxisGeometry vmecpp::ComputeThreed1AxisGeometry(
const Sizes& s, const FourierBasisFastPoloidal& fourier_basis,
const VmecInternalResults& vmec_internal_results) {
Threed1AxisGeometry result;

result.raxis_symm = VectorXd::Zero(s.ntor + 1);
result.zaxis_symm = VectorXd::Zero(s.ntor + 1);
if (s.lasym) {
result.raxis_asym = VectorXd::Zero(s.ntor + 1);
result.zaxis_asym = VectorXd::Zero(s.ntor + 1);
}

const int jF = 0;
const int m = 0;
for (int n = 0; n <= s.ntor; ++n) {
const double basis_scaling =
fourier_basis.mscale[0] * fourier_basis.nscale[n];

double tz = basis_scaling;
if (!s.lthreed) {
tz = 0.0;
}

const int index = (jF * (s.ntor + 1) + n) * s.mpol + m;

result.raxis_symm[n] = basis_scaling * vmec_internal_results.rmncc(index);
if (s.lthreed) {
result.zaxis_symm[n] = -tz * vmec_internal_results.zmncs(index);
}
if (s.lasym) {
result.zaxis_asym[n] = basis_scaling * vmec_internal_results.zmncc(index);
if (s.lthreed) {
result.raxis_asym[n] = -tz * vmec_internal_results.rmncs(index);
}
}
}

return result;
}

vmecpp::Threed1Betas vmecpp::ComputeThreed1Betas(
const HandoverStorage& handover_storage,
const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
const Threed1GeometricAndMagneticQuantitiesIntermediate&
threed1_geomag_intermediate,
const Threed1GeometricAndMagneticQuantities& threed1_geomag) {
Threed1Betas result;

result.betatot = threed1_geomag.betatot;
result.betapol = threed1_geomag.betapol;
result.betator = threed1_geomag.betator;

result.rbtor = handover_storage.rBtor;
result.betaxis = threed1_first_table_intermediate.beta_axis;
result.betstr =
2.0 *
std::sqrt(threed1_geomag_intermediate.sump2 / threed1_geomag.volume_p) /
(threed1_geomag_intermediate.sumbtot / threed1_geomag.volume_p);

return result;
}

vmecpp::Threed1ShafranovIntegrals vmecpp::ComputeThreed1ShafranovIntegrals(
const Sizes& s, const FlowControl& fc,
const HandoverStorage& handover_storage,
const VmecInternalResults& vmec_internal_results,
const Threed1GeometricAndMagneticQuantitiesIntermediate&
threed1_geometric_magnetic_intermediate,
const Threed1GeometricAndMagneticQuantities& threed1_geomag,
VacuumPressureState vacuum_pressure_state) {
Threed1ShafranovIntegrals result;

std::vector<double> bpol2vac(s.nZnT, 0.0);
if (fc.lfreeb &&
vacuum_pressure_state >= vmecpp::VacuumPressureState::kActive) {
for (int l = 0; l < s.nThetaEff; ++l) {
for (int k = 0; k < s.nZeta; ++k) {

int idx_lk = l * s.nZeta + k;
int idx_kl = k * s.nThetaEff + l;
const double bphiv = handover_storage.vacuum_b_phi[idx_lk];
bpol2vac[idx_kl] =
2.0 * handover_storage.vacuum_magnetic_pressure[idx_lk] -
bphiv * bphiv;
}
}
} else {
for (int kl = 0; kl < s.nZnT; ++kl) {
const int jH_ns2 = (fc.ns - 2) * s.nZnT + kl;
const int jH_ns3 = (fc.ns - 3) * s.nZnT + kl;

const double bsq_lcfs =
1.5 * vmec_internal_results.total_pressure(jH_ns2) -
0.5 * vmec_internal_results.total_pressure(jH_ns3);

const double bsupv_lcfs = 1.5 * vmec_internal_results.bsupv(jH_ns2) -
0.5 * vmec_internal_results.bsupv(jH_ns3);
const double btor_lfcs =
bsupv_lcfs * threed1_geometric_magnetic_intermediate.redge[kl];

bpol2vac[kl] = 2.0 * bsq_lcfs - btor_lfcs * btor_lfcs;
}
}

const double aminr1 = threed1_geomag.volume_p / threed1_geomag.surf_area_p;

double bpol2vac_surf_area_sum = 0.0;
for (int kl = 0; kl < s.nZnT; ++kl) {
bpol2vac_surf_area_sum +=
bpol2vac[kl] * threed1_geometric_magnetic_intermediate.surf_area[kl];
}
const double factor =
1.0 / (4.0 * M_PI * M_PI * aminr1 * bpol2vac_surf_area_sum);
const double facnorm = factor * 4.0 * M_PI * M_PI;

const double toroidal_current_per_circumference =
vmec_internal_results.currv / threed1_geomag.circum_p;
result.scaling_ratio = factor * toroidal_current_per_circumference *
toroidal_current_per_circumference *
threed1_geomag.volume_p;

double sigr0 = 0.0;
double sigr1 = 0.0;
double sigz1 = 0.0;
for (int kl = 0; kl < s.nZnT; ++kl) {
const int l = kl % s.nThetaEff;
const int lcfs_kl = (fc.ns - 1) * s.nZnT + kl;

const double r_edge = threed1_geometric_magnetic_intermediate.redge[kl];
const double z_edge =
vmec_internal_results.z_e(lcfs_kl) + vmec_internal_results.z_o(lcfs_kl);

const double p_hat = threed1_geometric_magnetic_intermediate.phat[kl];
const double rbps1u = facnorm * r_edge * p_hat * s.wInt[l];

sigr0 += rbps1u * vmec_internal_results.zuFull(lcfs_kl);
sigr1 += rbps1u * vmec_internal_results.zuFull(lcfs_kl) * r_edge;
sigz1 -= rbps1u * vmec_internal_results.ruFull(lcfs_kl) * z_edge;
}

const double er = sigr1 + sigz1;

const double rshaf = threed1_geometric_magnetic_intermediate.rshaf;
result.r_lao =
threed1_geomag.volume_p / (2.0 * M_PI * threed1_geomag.cross_area_p);
result.f_lao = rshaf / result.r_lao;
result.f_geo = rshaf / threed1_geomag.rcen;

result.smaleli = factor * threed1_geometric_magnetic_intermediate.sumbpol;
result.betai = 2.0 * factor * threed1_geometric_magnetic_intermediate.sump;
result.musubi = threed1_geometric_magnetic_intermediate.vnorm * factor *
threed1_geometric_magnetic_intermediate.musubi;
result.lambda = 0.5 * result.smaleli + result.betai;

result.s11 = er - rshaf * sigr0;

result.s12 = er - threed1_geomag.rcen * sigr0;

result.s13 = er - result.r_lao * sigr0;
result.s2 = sigr0 * rshaf;

result.s3 = sigz1;

result.delta1 = 0.0;
result.delta2 = 1.0 - result.f_geo;
result.delta3 = 1.0 - result.f_lao;

return result;
}

vmecpp::WOutFileContents vmecpp::ComputeWOutFileContents(
const VmecINDATA& indata, const Sizes& s, const FourierBasisFastPoloidal& t,
const FlowControl& fc, const VmecConstants& constants,
const HandoverStorage& handover_storage, const std::string& mgrid_mode,
const std::vector<std::string>& coil_group_names,
VmecInternalResults& m_vmec_internal_results, const BSubSHalf& bsubs_half,
const BSubSFull& bsubs_full, const MercierFileContents& mercier,
const JxBOutFileContents& jxbout,
const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
const Threed1FirstTable& threed1_first_table,
const Threed1GeometricAndMagneticQuantities& threed1_geomag,
const Threed1AxisGeometry& threed1_axis, const Threed1Betas& threed1_betas,
const Threed1FreeBoundary& threed1_free_boundary, VmecStatus vmec_status,
int iter2) {

WOutFileContents wout;

wout.version_ = 8.52;

wout.input_extension = "";

wout.signgs = m_vmec_internal_results.sign_of_jacobian;

wout.gamma = indata.gamma;

wout.pcurr_type = indata.pcurr_type;
wout.pmass_type = indata.pmass_type;
wout.piota_type = indata.piota_type;

wout.am = NonEmptyVectorOr(indata.am, 0.0);
wout.am_aux_s = NonEmptyVectorOr(indata.am_aux_s, -1.0);
wout.am_aux_f = NonEmptyVectorOr(indata.am_aux_f, 0.0);
wout.ac = NonEmptyVectorOr(indata.ac, 0.0);
wout.ac_aux_s = NonEmptyVectorOr(indata.ac_aux_s, -1.0);
wout.ac_aux_f = NonEmptyVectorOr(indata.ac_aux_f, 0.0);
wout.ai = NonEmptyVectorOr(indata.ai, 0.0);
wout.ai_aux_s = NonEmptyVectorOr(indata.ai_aux_s, -1.0);
wout.ai_aux_f = NonEmptyVectorOr(indata.ai_aux_f, 0.0);

constexpr int kPreset = 21;
constexpr int kNdfmax = 101;

auto PadVector = [](Eigen::VectorXd& vec, int target_size, double pad_value) {
if (vec.size() < target_size) {
const int old_size = vec.size();
vec.conservativeResize(target_size);
vec.tail(target_size - old_size).setConstant(pad_value);
}
};

PadVector(wout.am, kPreset, 0.0);
PadVector(wout.ac, kPreset, 0.0);
PadVector(wout.ai, kPreset, 0.0);

PadVector(wout.am_aux_s, kNdfmax, -1.0);
PadVector(wout.am_aux_f, kNdfmax, 0.0);
PadVector(wout.ac_aux_s, kNdfmax, -1.0);
PadVector(wout.ac_aux_f, kNdfmax, 0.0);
PadVector(wout.ai_aux_s, kNdfmax, -1.0);
PadVector(wout.ai_aux_f, kNdfmax, 0.0);

wout.nfp = indata.nfp;
wout.mpol = indata.mpol;
wout.ntor = indata.ntor;
wout.lasym = indata.lasym;

wout.ns = fc.ns;
wout.ftolv = fc.ftolv;

wout.niter = iter2;

wout.lfreeb = indata.lfreeb;
wout.mgrid_file = indata.mgrid_file;

wout.nextcur = static_cast<int>(indata.extcur.size());
wout.extcur = indata.extcur;
wout.mgrid_mode = mgrid_mode;

wout.wb = handover_storage.magneticEnergy;
wout.wp = handover_storage.thermalEnergy;

wout.rmax_surf = threed1_geomag.rmax_surf;
wout.rmin_surf = threed1_geomag.rmin_surf;
wout.zmax_surf = threed1_geomag.zmax_surf;

wout.mnmax = s.mnmax;
wout.mnmax_nyq = s.mnmax_nyq;

if (vmec_status == VmecStatus::SUCCESSFUL_TERMINATION) {
wout.ier_flag = VmecStatusCode(VmecStatus::NORMAL_TERMINATION);
} else {
wout.ier_flag = VmecStatusCode(vmec_status);
}

wout.aspect = threed1_geomag.aspect;

wout.betatotal = threed1_betas.betatot;
wout.betapol = threed1_betas.betapol;
wout.betator = threed1_betas.betator;
wout.betaxis = threed1_betas.betaxis;

wout.b0 = threed1_geomag.b0;

wout.rbtor0 = handover_storage.rBtor0;
wout.rbtor = handover_storage.rBtor;

wout.IonLarmor = threed1_geomag.IonLarmor;
wout.volavgB = threed1_geomag.VolAvgB;

wout.ctor = handover_storage.cTor / MU_0;

wout.Aminor_p = threed1_geomag.Aminor_p;
wout.Rmajor_p = threed1_geomag.Rmajor_p;
wout.volume = threed1_geomag.volume_p;

wout.fsqr = fc.fsqr;
wout.fsqz = fc.fsqz;
wout.fsql = fc.fsql;

wout.force_residual_r = ToEigenVector(fc.force_residual_r);
wout.force_residual_z = ToEigenVector(fc.force_residual_z);
wout.force_residual_lambda = ToEigenVector(fc.force_residual_lambda);
wout.fsqt = wout.force_residual_r + wout.force_residual_z +
wout.force_residual_lambda;
wout.delbsq = ToEigenVector(fc.delbsq);

wout.restart_reason_timetrace =
Eigen::VectorXi::Zero(fc.restart_reasons.size());
for (Eigen::Index i = 0;
i < static_cast<Eigen::Index>(fc.restart_reasons.size()); ++i) {

wout.restart_reason_timetrace(i) = static_cast<int>(fc.restart_reasons[i]);
}
wout.itfsq = wout.fsqt.size();

wout.wdot = ToEigenVector(fc.mhd_energy);
if (wout.wdot.size() > 1) {

wout.wdot.tail(wout.wdot.size() - 1) =
((wout.wdot.tail(wout.wdot.size() - 1) -
wout.wdot.head(wout.wdot.size() - 1))
.cwiseQuotient(wout.wdot.tail(wout.wdot.size() - 1)))
.eval();
}

wout.iotaf = m_vmec_internal_results.iotaF;

wout.q_factor = VectorXd::Ones(fc.ns) * DBL_MAX;

wout.presf = VectorXd::Zero(fc.ns);
wout.phipf = VectorXd::Zero(fc.ns);
wout.chipf = VectorXd::Zero(fc.ns);
wout.jcuru = VectorXd::Zero(fc.ns);
wout.jcurv = VectorXd::Zero(fc.ns);

for (int jF = 0; jF < fc.ns; ++jF) {
if (wout.iotaf[jF] != 0.0) {
wout.q_factor[jF] = 1.0 / wout.iotaf[jF];
}
wout.presf[jF] = threed1_first_table_intermediate.presf[jF] / MU_0;
wout.phipf[jF] = m_vmec_internal_results.sign_of_jacobian * 2.0 * M_PI *
m_vmec_internal_results.phipF[jF];
wout.chipf[jF] = m_vmec_internal_results.sign_of_jacobian * 2.0 * M_PI *
m_vmec_internal_results.chipF[jF];
wout.jcuru[jF] = threed1_first_table_intermediate.jcuru[jF] / MU_0;
wout.jcurv[jF] = threed1_first_table_intermediate.jcurv[jF] / MU_0;
}
wout.phi = m_vmec_internal_results.phiF;
wout.chi = threed1_first_table_intermediate.chi;
wout.specw = m_vmec_internal_results.spectral_width;

wout.mass = VectorXd::Zero(fc.ns);
wout.pres = VectorXd::Zero(fc.ns);
for (int jH = 0; jH < fc.ns - 1; ++jH) {
wout.mass[jH + 1] = m_vmec_internal_results.massH[jH] / MU_0;
wout.pres[jH + 1] = m_vmec_internal_results.presH[jH] / MU_0;
}
wout.iotas = VectorXd::Zero(fc.ns);
wout.iotas.tail(fc.ns - 1) = m_vmec_internal_results.iotaH;
wout.beta_vol = VectorXd::Zero(fc.ns);
wout.beta_vol.tail(fc.ns - 1) = threed1_first_table_intermediate.beta_vol;
wout.buco = VectorXd::Zero(fc.ns);
wout.buco.tail(fc.ns - 1) = threed1_first_table_intermediate.bucoH;
wout.bvco = VectorXd::Zero(fc.ns);
wout.bvco.tail(fc.ns - 1) = threed1_first_table_intermediate.bvcoH;
wout.vp = VectorXd::Zero(fc.ns);
wout.vp.tail(fc.ns - 1) = m_vmec_internal_results.dVdsH;
wout.phips = VectorXd::Zero(fc.ns);
wout.phips.tail(fc.ns - 1) = m_vmec_internal_results.phipH;
wout.over_r = VectorXd::Zero(fc.ns);
wout.over_r.tail(fc.ns - 1) = threed1_first_table_intermediate.overr;

wout.jdotb = jxbout.jdotb;
wout.bdotb = jxbout.bdotb;
wout.bdotgradv = jxbout.bdotgradv;

wout.DMerc = mercier.DMerc;
wout.DShear = mercier.Dshear;
wout.DWell = mercier.Dwell;
wout.DCurr = mercier.Dcurr;
wout.DGeod = mercier.Dgeod;

wout.equif = threed1_first_table.radial_force;

const Eigen::VectorXd& vacuum_potential = handover_storage.vacuum_potential;
if (vacuum_potential.size() > 0) {
const int nf = s.ntor;
const int mf = s.mpol + 1;
const int mnpd = (2 * nf + 1) * (mf + 1);
wout.potvac = VectorXd::Zero(2 * mnpd);
if (vacuum_potential.size() <= wout.potvac.size()) {
wout.potvac.head(vacuum_potential.size()) = vacuum_potential;
}

wout.xmpot = Eigen::VectorXi::Zero(mnpd);
wout.xnpot = Eigen::VectorXi::Zero(mnpd);
for (int mn = 0; mn < mnpd; ++mn) {
wout.xmpot[mn] = mn % (mf + 1);
wout.xnpot[mn] = (mn / (mf + 1) - nf) * s.nfp;
}
}

wout.curlabel = coil_group_names;

wout.xm = t.xm;
wout.xn = t.xn;
wout.xm_nyq = t.xm_nyq;
wout.xn_nyq = t.xn_nyq;

wout.raxis_cc = threed1_axis.raxis_symm;
wout.zaxis_cs = threed1_axis.zaxis_symm;

std::vector<double> cosmui(s.nThetaReduced * (s.mnyq2 + 1));
for (int ml = 0; ml < s.nThetaReduced * (s.mnyq2 + 1); ++ml) {
cosmui[ml] = t.cosmui[ml];
}
if (s.mnyq != 0) {
for (int l = 0; l < s.nThetaReduced; ++l) {
const int ml = s.mnyq * s.nThetaReduced + l;
cosmui[ml] /= 2.0;
}
}

std::vector<double> cosnv((s.nnyq2 + 1) * s.nZeta);
for (int kn = 0; kn < (s.nnyq2 + 1) * s.nZeta; ++kn) {
cosnv[kn] = t.cosnv[kn];
}
if (s.nnyq != 0) {
for (int k = 0; k < s.nZeta; ++k) {

const int kn = k * (s.nnyq2 + 1) + s.nnyq;
cosnv[kn] /= 2.0;
}
}

const double sigma = -m_vmec_internal_results.sign_of_jacobian;
if (s.lthreed) {
for (int jF = 0; jF < fc.ns; ++jF) {
for (int n = 0; n < s.ntor + 1; ++n) {
const int m = 1;
const int idx_fc = (jF * (s.ntor + 1) + n) * s.mpol + m;

const double old_rss = m_vmec_internal_results.rmnss(idx_fc);
m_vmec_internal_results.rmnss(idx_fc) =
(old_rss + sigma * m_vmec_internal_results.zmncs(idx_fc));
m_vmec_internal_results.zmncs(idx_fc) =
(sigma * old_rss - m_vmec_internal_results.zmncs(idx_fc));
}
}
}

wout.rmnc = RowMatrixXd::Zero(s.mnmax, fc.ns);
wout.zmns = RowMatrixXd::Zero(s.mnmax, fc.ns);
wout.lmns_full = RowMatrixXd::Zero(s.mnmax, fc.ns);
for (int jF = 0; jF < fc.ns; ++jF) {
std::vector<double> rmnc1(s.mnmax, 0.0);
std::vector<double> zmns1(s.mnmax, 0.0);
std::vector<double> lmns1(s.mnmax, 0.0);

int mn = -1;
int m_0 = 0;
for (int n = 0; n <= s.ntor; ++n) {
mn++;
const int idx_fc = (jF * (s.ntor + 1) + n) * s.mpol + m_0;
const double t1 = t.mscale[m_0] * t.nscale[n];
rmnc1[mn] = t1 * m_vmec_internal_results.rmncc(idx_fc);
if (s.lthreed) {
zmns1[mn] = -t1 * m_vmec_internal_results.zmncs(idx_fc);
lmns1[mn] = -t1 * m_vmec_internal_results.lmncs(idx_fc);
}

}

if (s.lthreed && jF == 0) {
int mn = -1;
for (int n = 0; n <= s.ntor; ++n) {
mn++;
const int idx_ns_1 = (1 * (s.ntor + 1) + n) * s.mpol + m_0;
const int idx_ns_2 = (2 * (s.ntor + 1) + n) * s.mpol + m_0;
const double t1 = t.mscale[m_0] * t.nscale[n];
lmns1[mn] = -t1 * (2.0 * m_vmec_internal_results.lmncs(idx_ns_1) -
m_vmec_internal_results.lmncs(idx_ns_2));
}
}

for (int m = 1; m < s.mpol; ++m) {
for (int n = -s.ntor; n <= s.ntor; ++n) {
mn++;
const int abs_n = std::abs(n);
const int idx_fc = (jF * (s.ntor + 1) + abs_n) * s.mpol + m;
const double t1 = t.mscale[m] * t.nscale[abs_n];
if (n == 0) {
rmnc1[mn] = t1 * m_vmec_internal_results.rmncc(idx_fc);
zmns1[mn] = t1 * m_vmec_internal_results.zmnsc(idx_fc);
lmns1[mn] = t1 * m_vmec_internal_results.lmnsc(idx_fc);
} else if (jF > 0) {
rmnc1[mn] = t1 * m_vmec_internal_results.rmncc(idx_fc) / 2.0;
zmns1[mn] = t1 * m_vmec_internal_results.zmnsc(idx_fc) / 2.0;
lmns1[mn] = t1 * m_vmec_internal_results.lmnsc(idx_fc) / 2.0;
if (s.lthreed) {
const int sign_n = signum(n);
rmnc1[mn] +=
t1 * sign_n * m_vmec_internal_results.rmnss(idx_fc) / 2.0;
zmns1[mn] -=
t1 * sign_n * m_vmec_internal_results.zmncs(idx_fc) / 2.0;
lmns1[mn] -=
t1 * sign_n * m_vmec_internal_results.lmncs(idx_fc) / 2.0;
}
}

}
}

CHECK_EQ(mn + 1, s.mnmax) << "counting error: (mn + 1)=" << (mn + 1)
<< " should be mnmax=" << s.mnmax;

for (int mn = 0; mn < s.mnmax; ++mn) {
wout.rmnc(mn, jF) = rmnc1[mn];
wout.zmns(mn, jF) = zmns1[mn];
wout.lmns_full(mn, jF) =
lmns1[mn] / m_vmec_internal_results.phipF[jF] * constants.lamscale;
}
}

wout.lmns = RowMatrixXd::Zero(s.mnmax, fc.ns);
for (int jH = 0; jH < fc.ns - 1; ++jH) {
const int jFi = jH;
const int jFo = jH + 1;

for (int mn = 0; mn < s.mnmax; ++mn) {
const double lmns_outside = wout.lmns_full(mn, jFo);

double lmns_inside = wout.lmns_full(mn, jFi);
if (jFi == 0 && wout.xm[mn] <= 1) {
lmns_inside = lmns_outside;
}

if (wout.xm[mn] % 2 == 0) {

wout.lmns(mn, jH + 1) = (lmns_outside + lmns_inside) / 2.0;
} else {

const double sm = m_vmec_internal_results.sm[jH];
const double sp = m_vmec_internal_results.sp[jH];
wout.lmns(mn, jH + 1) = (sm * lmns_outside + sp * lmns_inside) / 2.0;
}
}
}

std::vector<double> magnetic_pressure((fc.ns - 1) * s.nZnT, 0.0);
#ifdef _OPENMP
#pragma omp parallel for num_threads(fc.max_threads())
#endif
for (int jH = 0; jH < fc.ns - 1; ++jH) {
for (int kl = 0; kl < s.nZnT; ++kl) {
const int idx_kl = jH * s.nZnT + kl;
const double total_pressure =
m_vmec_internal_results.total_pressure(idx_kl);
magnetic_pressure[idx_kl] = std::sqrt(
2.0 * std::abs(total_pressure - m_vmec_internal_results.presH[jH]));
}
}

const double tmult = 0.5;

wout.gmnc = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.bmnc = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.bsubumnc = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.bsubvmnc = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);

wout.bsubsmns = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);

wout.bsupumnc = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.bsupvmnc = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);

if (s.lasym) {
wout.gmns = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.bmns = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.bsubumns = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.bsubvmns = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.bsubsmnc = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.bsupumns = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.bsupvmns = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
}

const int partial_sum_size = (s.mnyq + 1) * s.nZeta;

#ifdef _OPENMP
#pragma omp parallel num_threads(fc.max_threads())
{
#endif
std::vector<double> Fc_gsqrt(partial_sum_size), Fs_gsqrt(partial_sum_size),
Fc_bmnc(partial_sum_size), Fs_bmnc(partial_sum_size),
Fc_bsubu(partial_sum_size), Fs_bsubu(partial_sum_size),
Fc_bsubv(partial_sum_size), Fs_bsubv(partial_sum_size),
Fc_bsupu(partial_sum_size), Fs_bsupu(partial_sum_size),
Fc_bsupv(partial_sum_size), Fs_bsupv(partial_sum_size),
Fc_bsubs(partial_sum_size), Fs_bsubs(partial_sum_size);

std::vector<double> Fc_gsqrt_a, Fs_gsqrt_a;
std::vector<double> Fc_bmnc_a, Fs_bmnc_a;
std::vector<double> Fc_bsubu_a, Fs_bsubu_a;
std::vector<double> Fc_bsubv_a, Fs_bsubv_a;
std::vector<double> Fc_bsupu_a, Fs_bsupu_a;
std::vector<double> Fc_bsupv_a, Fs_bsupv_a;
std::vector<double> Fc_bsubs_a, Fs_bsubs_a;
if (s.lasym) {
Fc_gsqrt_a.resize(partial_sum_size);
Fs_gsqrt_a.resize(partial_sum_size);
Fc_bmnc_a.resize(partial_sum_size);
Fs_bmnc_a.resize(partial_sum_size);
Fc_bsubu_a.resize(partial_sum_size);
Fs_bsubu_a.resize(partial_sum_size);
Fc_bsubv_a.resize(partial_sum_size);
Fs_bsubv_a.resize(partial_sum_size);
Fc_bsupu_a.resize(partial_sum_size);
Fs_bsupu_a.resize(partial_sum_size);
Fc_bsupv_a.resize(partial_sum_size);
Fs_bsupv_a.resize(partial_sum_size);
Fc_bsubs_a.resize(partial_sum_size);
Fs_bsubs_a.resize(partial_sum_size);
}

#ifdef _OPENMP
#pragma omp for
#endif
for (int jH = 0; jH < fc.ns - 1; ++jH) {

for (int m = 0; m <= s.mnyq; ++m) {
const int m_nzeta_p1 = m * s.nZeta;
for (int k = 0; k < s.nZeta; ++k) {
double fc_gsqrt = 0.0, fs_gsqrt = 0.0;
double fc_bmnc = 0.0, fs_bmnc = 0.0;
double fc_bsubu = 0.0, fs_bsubu = 0.0;
double fc_bsubv = 0.0, fs_bsubv = 0.0;
double fc_bsupu = 0.0, fs_bsupu = 0.0;
double fc_bsupv = 0.0, fs_bsupv = 0.0;
double fc_bsubs = 0.0, fs_bsubs = 0.0;

int k_rev = 0;
double fc_gsqrt_a = 0.0, fs_gsqrt_a = 0.0;
double fc_bmnc_a = 0.0, fs_bmnc_a = 0.0;
double fc_bsubu_a = 0.0, fs_bsubu_a = 0.0;
double fc_bsubv_a = 0.0, fs_bsubv_a = 0.0;
double fc_bsupu_a = 0.0, fs_bsupu_a = 0.0;
double fc_bsupv_a = 0.0, fs_bsupv_a = 0.0;
double fc_bsubs_a = 0.0, fs_bsubs_a = 0.0;
if (s.lasym) {
k_rev = (s.nZeta - k) % s.nZeta;
}

for (int l = 0; l < s.nThetaReduced; ++l) {
const int ml = m * s.nThetaReduced + l;
const int idx_kl = (jH * s.nZeta + k) * s.nThetaEff + l;
const double cmu = cosmui[ml];
const double smu = t.sinmui[ml];

double g = m_vmec_internal_results.gsqrt(idx_kl);
double mp = magnetic_pressure[idx_kl];
double bu = m_vmec_internal_results.bsubu(idx_kl);
double bv = m_vmec_internal_results.bsubv(idx_kl);
double bpu = m_vmec_internal_results.bsupu(idx_kl);
double bpv = m_vmec_internal_results.bsupv(idx_kl);
double bs = bsubs_half.bsubs_half(idx_kl);

if (s.lasym) {

const int l_rev = (s.nThetaEff - l) % s.nThetaEff;
const int idx_kl_rev =
(jH * s.nZeta + k_rev) * s.nThetaEff + l_rev;

const double g_rev = m_vmec_internal_results.gsqrt(idx_kl_rev);
const double mp_rev = magnetic_pressure[idx_kl_rev];
const double bu_rev = m_vmec_internal_results.bsubu(idx_kl_rev);
const double bv_rev = m_vmec_internal_results.bsubv(idx_kl_rev);
const double bpu_rev = m_vmec_internal_results.bsupu(idx_kl_rev);
const double bpv_rev = m_vmec_internal_results.bsupv(idx_kl_rev);
const double bs_rev = bsubs_half.bsubs_half(idx_kl_rev);

const double g_a = 0.5 * (g - g_rev);
const double mp_a = 0.5 * (mp - mp_rev);
const double bu_a = 0.5 * (bu - bu_rev);
const double bv_a = 0.5 * (bv - bv_rev);
const double bpu_a = 0.5 * (bpu - bpu_rev);
const double bpv_a = 0.5 * (bpv - bpv_rev);

const double bs_a = 0.5 * (bs + bs_rev);

g = 0.5 * (g + g_rev);
mp = 0.5 * (mp + mp_rev);
bu = 0.5 * (bu + bu_rev);
bv = 0.5 * (bv + bv_rev);
bpu = 0.5 * (bpu + bpu_rev);
bpv = 0.5 * (bpv + bpv_rev);

bs = 0.5 * (bs - bs_rev);

fc_gsqrt_a += cmu * g_a;
fs_gsqrt_a += smu * g_a;
fc_bmnc_a += cmu * mp_a;
fs_bmnc_a += smu * mp_a;
fc_bsubu_a += cmu * bu_a;
fs_bsubu_a += smu * bu_a;
fc_bsubv_a += cmu * bv_a;
fs_bsubv_a += smu * bv_a;
fc_bsupu_a += cmu * bpu_a;
fs_bsupu_a += smu * bpu_a;
fc_bsupv_a += cmu * bpv_a;
fs_bsupv_a += smu * bpv_a;
fc_bsubs_a += cmu * bs_a;
fs_bsubs_a += smu * bs_a;
}

fc_gsqrt += cmu * g;
fs_gsqrt += smu * g;
fc_bmnc += cmu * mp;
fs_bmnc += smu * mp;
fc_bsubu += cmu * bu;
fs_bsubu += smu * bu;
fc_bsubv += cmu * bv;
fs_bsubv += smu * bv;
fc_bsupu += cmu * bpu;
fs_bsupu += smu * bpu;
fc_bsupv += cmu * bpv;
fs_bsupv += smu * bpv;
fc_bsubs += cmu * bs;
fs_bsubs += smu * bs;
}

const int idx_mk = m_nzeta_p1 + k;
Fc_gsqrt[idx_mk] = fc_gsqrt;
Fs_gsqrt[idx_mk] = fs_gsqrt;
Fc_bmnc[idx_mk] = fc_bmnc;
Fs_bmnc[idx_mk] = fs_bmnc;
Fc_bsubu[idx_mk] = fc_bsubu;
Fs_bsubu[idx_mk] = fs_bsubu;
Fc_bsubv[idx_mk] = fc_bsubv;
Fs_bsubv[idx_mk] = fs_bsubv;
Fc_bsupu[idx_mk] = fc_bsupu;
Fs_bsupu[idx_mk] = fs_bsupu;
Fc_bsupv[idx_mk] = fc_bsupv;
Fs_bsupv[idx_mk] = fs_bsupv;
Fc_bsubs[idx_mk] = fc_bsubs;
Fs_bsubs[idx_mk] = fs_bsubs;
if (s.lasym) {
Fc_gsqrt_a[idx_mk] = fc_gsqrt_a;
Fs_gsqrt_a[idx_mk] = fs_gsqrt_a;
Fc_bmnc_a[idx_mk] = fc_bmnc_a;
Fs_bmnc_a[idx_mk] = fs_bmnc_a;
Fc_bsubu_a[idx_mk] = fc_bsubu_a;
Fs_bsubu_a[idx_mk] = fs_bsubu_a;
Fc_bsubv_a[idx_mk] = fc_bsubv_a;
Fs_bsubv_a[idx_mk] = fs_bsubv_a;
Fc_bsupu_a[idx_mk] = fc_bsupu_a;
Fs_bsupu_a[idx_mk] = fs_bsupu_a;
Fc_bsupv_a[idx_mk] = fc_bsupv_a;
Fs_bsupv_a[idx_mk] = fs_bsupv_a;
Fc_bsubs_a[idx_mk] = fc_bsubs_a;
Fs_bsubs_a[idx_mk] = fs_bsubs_a;
}
}
}

for (int mn_nyq = 0; mn_nyq < s.mnmax_nyq; ++mn_nyq) {
const int m = wout.xm_nyq[mn_nyq];
const int n = wout.xn_nyq[mn_nyq] / wout.nfp;
const int abs_n = std::abs(n);
const int sign_n = signum(n);
double dmult = t.mscale[m] * t.nscale[abs_n] * tmult;
if (m == 0 || n == 0) {
dmult *= 2.0;
}
const int m_nzeta = m * s.nZeta;

double acc_gmnc = 0.0, acc_bmnc = 0.0, acc_bsubumnc = 0.0,
acc_bsubvmnc = 0.0, acc_bsubsmns = 0.0, acc_bsupumnc = 0.0,
acc_bsupvmnc = 0.0;

for (int k = 0; k < s.nZeta; ++k) {
const int kn = k * (s.nnyq2 + 1) + abs_n;
const double cnv = cosnv[kn];
const double snv = t.sinnv[kn];
const int idx_mk = m_nzeta + k;

acc_gmnc += cnv * Fc_gsqrt[idx_mk] + sign_n * snv * Fs_gsqrt[idx_mk];
acc_bmnc += cnv * Fc_bmnc[idx_mk] + sign_n * snv * Fs_bmnc[idx_mk];
acc_bsubumnc +=
cnv * Fc_bsubu[idx_mk] + sign_n * snv * Fs_bsubu[idx_mk];
acc_bsubvmnc +=
cnv * Fc_bsubv[idx_mk] + sign_n * snv * Fs_bsubv[idx_mk];
acc_bsupumnc +=
cnv * Fc_bsupu[idx_mk] + sign_n * snv * Fs_bsupu[idx_mk];
acc_bsupvmnc +=
cnv * Fc_bsupv[idx_mk] + sign_n * snv * Fs_bsupv[idx_mk];

acc_bsubsmns +=
cnv * Fs_bsubs[idx_mk] - sign_n * snv * Fc_bsubs[idx_mk];
}

wout.gmnc(mn_nyq, jH + 1) = dmult * acc_gmnc;
wout.bmnc(mn_nyq, jH + 1) = dmult * acc_bmnc;
wout.bsubumnc(mn_nyq, jH + 1) = dmult * acc_bsubumnc;
wout.bsubvmnc(mn_nyq, jH + 1) = dmult * acc_bsubvmnc;
wout.bsubsmns(mn_nyq, jH + 1) = dmult * acc_bsubsmns;
wout.bsupumnc(mn_nyq, jH + 1) = dmult * acc_bsupumnc;
wout.bsupvmnc(mn_nyq, jH + 1) = dmult * acc_bsupvmnc;

if (s.lasym) {
double acc_gmns = 0.0;
double acc_bmns = 0.0;
double acc_bsubumns = 0.0;
double acc_bsubvmns = 0.0;
double acc_bsubsmnc = 0.0;
double acc_bsupumns = 0.0;
double acc_bsupvmns = 0.0;

for (int k = 0; k < s.nZeta; ++k) {
const int kn = k * (s.nnyq2 + 1) + abs_n;
const double cnv = cosnv[kn];
const double snv = t.sinnv[kn];
const int idx_mk = m_nzeta + k;

acc_gmns +=
cnv * Fs_gsqrt_a[idx_mk] - sign_n * snv * Fc_gsqrt_a[idx_mk];
acc_bmns +=
cnv * Fs_bmnc_a[idx_mk] - sign_n * snv * Fc_bmnc_a[idx_mk];
acc_bsubumns +=
cnv * Fs_bsubu_a[idx_mk] - sign_n * snv * Fc_bsubu_a[idx_mk];
acc_bsubvmns +=
cnv * Fs_bsubv_a[idx_mk] - sign_n * snv * Fc_bsubv_a[idx_mk];
acc_bsupumns +=
cnv * Fs_bsupu_a[idx_mk] - sign_n * snv * Fc_bsupu_a[idx_mk];
acc_bsupvmns +=
cnv * Fs_bsupv_a[idx_mk] - sign_n * snv * Fc_bsupv_a[idx_mk];

acc_bsubsmnc +=
cnv * Fc_bsubs_a[idx_mk] + sign_n * snv * Fs_bsubs_a[idx_mk];
}

wout.gmns(mn_nyq, jH + 1) = dmult * acc_gmns;
wout.bmns(mn_nyq, jH + 1) = dmult * acc_bmns;
wout.bsubumns(mn_nyq, jH + 1) = dmult * acc_bsubumns;
wout.bsubvmns(mn_nyq, jH + 1) = dmult * acc_bsubvmns;
wout.bsubsmnc(mn_nyq, jH + 1) = dmult * acc_bsubsmnc;
wout.bsupumns(mn_nyq, jH + 1) = dmult * acc_bsupumns;
wout.bsupvmns(mn_nyq, jH + 1) = dmult * acc_bsupvmns;
}
}
}

#ifdef _OPENMP
}
#endif

for (int mn_nyq = 0; mn_nyq < s.mnmax_nyq; ++mn_nyq) {
wout.bsubsmns(mn_nyq, 0) =
2.0 * wout.bsubsmns(mn_nyq, 1) - wout.bsubsmns(mn_nyq, 2);
}
if (s.lasym) {
for (int mn_nyq = 0; mn_nyq < s.mnmax_nyq; ++mn_nyq) {
wout.bsubsmnc(mn_nyq, 0) =
2.0 * wout.bsubsmnc(mn_nyq, 1) - wout.bsubsmnc(mn_nyq, 2);
}
}

wout.bsubsmns_full = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
if (s.lasym) {
wout.bsubsmnc_full = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
}

#ifdef _OPENMP
#pragma omp parallel num_threads(fc.max_threads())
{
#endif
std::vector<double> Fc_bsubs_full(partial_sum_size),
Fs_bsubs_full(partial_sum_size);

std::vector<double> Fc_bsubs_full_a, Fs_bsubs_full_a;
if (s.lasym) {
Fc_bsubs_full_a.resize(partial_sum_size);
Fs_bsubs_full_a.resize(partial_sum_size);
}

#ifdef _OPENMP
#pragma omp for
#endif
for (int jF = 0; jF < fc.ns; ++jF) {
std::fill(Fc_bsubs_full.begin(), Fc_bsubs_full.end(), 0.0);
std::fill(Fs_bsubs_full.begin(), Fs_bsubs_full.end(), 0.0);
if (s.lasym) {
std::fill(Fc_bsubs_full_a.begin(), Fc_bsubs_full_a.end(), 0.0);
std::fill(Fs_bsubs_full_a.begin(), Fs_bsubs_full_a.end(), 0.0);
}

for (int m = 0; m <= s.mnyq; ++m) {
const int m_nzeta = m * s.nZeta;
for (int k = 0; k < s.nZeta; ++k) {
double fc = 0.0, fs = 0.0;
double fc_a = 0.0, fs_a = 0.0;
int k_rev = 0;
if (s.lasym) {
k_rev = (s.nZeta - k) % s.nZeta;
}
for (int l = 0; l < s.nThetaReduced; ++l) {
const int ml = m * s.nThetaReduced + l;
const int idx_kl = (jF * s.nZeta + k) * s.nThetaEff + l;
double bs = bsubs_full.bsubs_full(idx_kl);
if (s.lasym) {

const int l_rev = (s.nThetaEff - l) % s.nThetaEff;
const int idx_kl_rev =
(jF * s.nZeta + k_rev) * s.nThetaEff + l_rev;
const double bs_rev = bsubs_full.bsubs_full(idx_kl_rev);
const double bs_a = 0.5 * (bs + bs_rev);
bs = 0.5 * (bs - bs_rev);
fc_a += cosmui[ml] * bs_a;
fs_a += t.sinmui[ml] * bs_a;
}
fc += cosmui[ml] * bs;
fs += t.sinmui[ml] * bs;
}
Fc_bsubs_full[m_nzeta + k] = fc;
Fs_bsubs_full[m_nzeta + k] = fs;
if (s.lasym) {
Fc_bsubs_full_a[m_nzeta + k] = fc_a;
Fs_bsubs_full_a[m_nzeta + k] = fs_a;
}
}
}

for (int mn_nyq = 0; mn_nyq < s.mnmax_nyq; ++mn_nyq) {
const int m = wout.xm_nyq[mn_nyq];
const int n = wout.xn_nyq[mn_nyq] / wout.nfp;
const int abs_n = std::abs(n);
const int sign_n = signum(n);
double dmult = t.mscale[m] * t.nscale[abs_n] * tmult;
if (m == 0 || n == 0) {
dmult *= 2.0;
}
const int m_nzeta = m * s.nZeta;

double acc = 0.0;
double acc_a = 0.0;
for (int k = 0; k < s.nZeta; ++k) {
const int kn = k * (s.nnyq2 + 1) + abs_n;
const int idx_mk = m_nzeta + k;

acc += cosnv[kn] * Fs_bsubs_full[idx_mk] -
sign_n * t.sinnv[kn] * Fc_bsubs_full[idx_mk];
if (s.lasym) {

acc_a += cosnv[kn] * Fc_bsubs_full_a[idx_mk] +
sign_n * t.sinnv[kn] * Fs_bsubs_full_a[idx_mk];
}
}
wout.bsubsmns_full(mn_nyq, jF) = dmult * acc;
if (s.lasym) {
wout.bsubsmnc_full(mn_nyq, jF) = dmult * acc_a;
}
}
}

#ifdef _OPENMP
}
#endif

if (s.lasym) {
wout.raxis_cs = threed1_axis.raxis_asym;
wout.zaxis_cc = threed1_axis.zaxis_asym;

for (int jF = 0; jF < fc.ns; ++jF) {
for (int n = 0; n < s.ntor + 1; ++n) {
const int m = 1;
const int idx_fc = (jF * (s.ntor + 1) + n) * s.mpol + m;

const double old_rsc = m_vmec_internal_results.rmnsc(idx_fc);
m_vmec_internal_results.rmnsc(idx_fc) =
(old_rsc + sigma * m_vmec_internal_results.zmncc(idx_fc));
m_vmec_internal_results.zmncc(idx_fc) =
(sigma * old_rsc - m_vmec_internal_results.zmncc(idx_fc));
}
}

wout.rmns = RowMatrixXd::Zero(s.mnmax, fc.ns);
wout.zmnc = RowMatrixXd::Zero(s.mnmax, fc.ns);
wout.lmnc_full = RowMatrixXd::Zero(s.mnmax, fc.ns);
for (int jF = 0; jF < fc.ns; ++jF) {
std::vector<double> rmns1(s.mnmax, 0.0);
std::vector<double> zmnc1(s.mnmax, 0.0);
std::vector<double> lmnc1(s.mnmax, 0.0);

int mn = -1;
int m_0 = 0;
for (int n = 0; n <= s.ntor; ++n) {
mn++;
const int idx_fc = (jF * (s.ntor + 1) + n) * s.mpol + m_0;
const double t1 = t.mscale[m_0] * t.nscale[n];
zmnc1[mn] = t1 * m_vmec_internal_results.zmncc(idx_fc);
lmnc1[mn] = t1 * m_vmec_internal_results.lmncc(idx_fc);
if (s.lthreed) {
rmns1[mn] = -t1 * m_vmec_internal_results.rmncs(idx_fc);
}

}

for (int m = 1; m < s.mpol; ++m) {
for (int n = -s.ntor; n <= s.ntor; ++n) {
mn++;
const int abs_n = std::abs(n);
const int idx_fc = (jF * (s.ntor + 1) + abs_n) * s.mpol + m;
const double t1 = t.mscale[m] * t.nscale[abs_n];
if (n == 0) {
rmns1[mn] = t1 * m_vmec_internal_results.rmnsc(idx_fc);
zmnc1[mn] = t1 * m_vmec_internal_results.zmncc(idx_fc);
lmnc1[mn] = t1 * m_vmec_internal_results.lmncc(idx_fc);
} else if (jF > 0) {
rmns1[mn] = t1 * m_vmec_internal_results.rmnsc(idx_fc) / 2.0;
zmnc1[mn] = t1 * m_vmec_internal_results.zmncc(idx_fc) / 2.0;
lmnc1[mn] = t1 * m_vmec_internal_results.lmncc(idx_fc) / 2.0;
if (s.lthreed) {
const int sign_n = signum(n);
rmns1[mn] -=
t1 * sign_n * m_vmec_internal_results.rmncs(idx_fc) / 2.0;
zmnc1[mn] +=
t1 * sign_n * m_vmec_internal_results.zmnss(idx_fc) / 2.0;
lmnc1[mn] +=
t1 * sign_n * m_vmec_internal_results.lmnss(idx_fc) / 2.0;
}
}

}
}

CHECK_EQ(mn + 1, s.mnmax) << "counting error: (mn + 1)=" << (mn + 1)
<< " should be mnmax=" << s.mnmax;

for (int mn = 0; mn < s.mnmax; ++mn) {
wout.rmns(mn, jF) = rmns1[mn];
wout.zmnc(mn, jF) = zmnc1[mn];
wout.lmnc_full(mn, jF) =
lmnc1[mn] / m_vmec_internal_results.phipF[jF] * constants.lamscale;
}
}

wout.lmnc = RowMatrixXd::Zero(s.mnmax, fc.ns);
for (int jH = 0; jH < fc.ns - 1; ++jH) {
const int jFi = jH;
const int jFo = jH + 1;

for (int mn = 0; mn < s.mnmax; ++mn) {
const double lmnc_outside = wout.lmnc_full(mn, jFo);

double lmnc_inside = wout.lmnc_full(mn, jFi);
if (jFi == 0 && wout.xm[mn] <= 1) {
lmnc_inside = lmnc_outside;
}

if (wout.xm[mn] % 2 == 0) {

wout.lmnc(mn, jH + 1) = (lmnc_outside + lmnc_inside) / 2.0;
} else {

const double sm = m_vmec_internal_results.sm[jH];
const double sp = m_vmec_internal_results.sp[jH];
wout.lmnc(mn, jH + 1) = (sm * lmnc_outside + sp * lmnc_inside) / 2.0;
}
}
}
}

{
const double ohs = 1.0 / fc.deltaS;

wout.currumnc = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.currvmnc = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
if (s.lasym) {
wout.currumns = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
wout.currvmns = RowMatrixXd::Zero(s.mnmax_nyq, fc.ns);
}

for (int j_f = 1; j_f < fc.ns - 1; ++j_f) {
const double sqrt_s_half_inner = m_vmec_internal_results.sqrtSH[j_f - 1];
const double sqrt_s_half_outer = m_vmec_internal_results.sqrtSH[j_f];
const double sqrt_s_full = m_vmec_internal_results.sqrtSF[j_f];

for (int mn = 0; mn < s.mnmax_nyq; ++mn) {
const int m = wout.xm_nyq[mn];
const double n_nfp = static_cast<double>(wout.xn_nyq[mn]);

double t1 = 0.0;
double t2 = 0.0;
double t3 = 0.0;

if (m % 2 == 1) {

t1 = 0.5 *
(sqrt_s_half_outer * wout.bsubsmns(mn, j_f + 1) +
sqrt_s_half_inner * wout.bsubsmns(mn, j_f)) /
sqrt_s_full;

const double bu0 = wout.bsubumnc(mn, j_f) / sqrt_s_half_inner;
const double bu1 = wout.bsubumnc(mn, j_f + 1) / sqrt_s_half_outer;
t2 = ohs * (bu1 - bu0) * sqrt_s_full +
0.25 * (bu0 + bu1) / sqrt_s_full;

const double bv0 = wout.bsubvmnc(mn, j_f) / sqrt_s_half_inner;
const double bv1 = wout.bsubvmnc(mn, j_f + 1) / sqrt_s_half_outer;
t3 = ohs * (bv1 - bv0) * sqrt_s_full +
0.25 * (bv0 + bv1) / sqrt_s_full;
} else {

t1 = 0.5 * (wout.bsubsmns(mn, j_f + 1) + wout.bsubsmns(mn, j_f));
t2 = ohs * (wout.bsubumnc(mn, j_f + 1) - wout.bsubumnc(mn, j_f));
t3 = ohs * (wout.bsubvmnc(mn, j_f + 1) - wout.bsubvmnc(mn, j_f));
}

wout.currumnc(mn, j_f) = -n_nfp * t1 - t3;
wout.currvmnc(mn, j_f) = -m * t1 + t2;

if (s.lasym) {
double t1a = 0.0;
double t2a = 0.0;
double t3a = 0.0;

if (m % 2 == 1) {
t1a = 0.5 *
(sqrt_s_half_outer * wout.bsubsmnc(mn, j_f + 1) +
sqrt_s_half_inner * wout.bsubsmnc(mn, j_f)) /
sqrt_s_full;

const double bu0a = wout.bsubumns(mn, j_f) / sqrt_s_half_inner;
const double bu1a = wout.bsubumns(mn, j_f + 1) / sqrt_s_half_outer;
t2a = ohs * (bu1a - bu0a) * sqrt_s_full +
0.25 * (bu0a + bu1a) / sqrt_s_full;

const double bv0a = wout.bsubvmns(mn, j_f) / sqrt_s_half_inner;
const double bv1a = wout.bsubvmns(mn, j_f + 1) / sqrt_s_half_outer;
t3a = ohs * (bv1a - bv0a) * sqrt_s_full +
0.25 * (bv0a + bv1a) / sqrt_s_full;
} else {
t1a = 0.5 * (wout.bsubsmnc(mn, j_f + 1) + wout.bsubsmnc(mn, j_f));
t2a = ohs * (wout.bsubumns(mn, j_f + 1) - wout.bsubumns(mn, j_f));
t3a = ohs * (wout.bsubvmns(mn, j_f + 1) - wout.bsubvmns(mn, j_f));
}

wout.currumns(mn, j_f) = n_nfp * t1a - t3a;
wout.currvmns(mn, j_f) = m * t1a + t2a;
}
}
}

for (int mn = 0; mn < s.mnmax_nyq; ++mn) {
if (wout.xm_nyq[mn] <= 1) {
wout.currumnc(mn, 0) =
2.0 * wout.currumnc(mn, 1) - wout.currumnc(mn, 2);
wout.currvmnc(mn, 0) =
2.0 * wout.currvmnc(mn, 1) - wout.currvmnc(mn, 2);
}

}

for (int mn = 0; mn < s.mnmax_nyq; ++mn) {
wout.currumnc(mn, fc.ns - 1) =
2.0 * wout.currumnc(mn, fc.ns - 2) - wout.currumnc(mn, fc.ns - 3);
wout.currvmnc(mn, fc.ns - 1) =
2.0 * wout.currvmnc(mn, fc.ns - 2) - wout.currvmnc(mn, fc.ns - 3);
}

wout.currumnc /= MU_0;
wout.currvmnc /= MU_0;

if (s.lasym) {
for (int mn = 0; mn < s.mnmax_nyq; ++mn) {
if (wout.xm_nyq[mn] <= 1) {
wout.currumns(mn, 0) =
2.0 * wout.currumns(mn, 1) - wout.currumns(mn, 2);
wout.currvmns(mn, 0) =
2.0 * wout.currvmns(mn, 1) - wout.currvmns(mn, 2);
}
}

for (int mn = 0; mn < s.mnmax_nyq; ++mn) {
wout.currumns(mn, fc.ns - 1) =
2.0 * wout.currumns(mn, fc.ns - 2) - wout.currumns(mn, fc.ns - 3);
wout.currvmns(mn, fc.ns - 1) =
2.0 * wout.currvmns(mn, fc.ns - 2) - wout.currvmns(mn, fc.ns - 3);
}

wout.currumns /= MU_0;
wout.currvmns /= MU_0;
}
}

return wout;
}

// source: vmecpp/vmec/profile_parameterization_data/profile_parameterization_data.cc

#include <string>
#include <vector>

namespace vmecpp {
namespace {

std::vector<ProfileParameterizationData> BuildProfileParameterizations() {

std::vector<ProfileParameterizationData> all;
all.reserve(NUM_PARAM);
all.emplace_back("---invalid---",  false,
false,   false,
false);
all.emplace_back("power_series",  true,
true,   true,
false);
all.emplace_back("power_series_i",  false,
true,   false,
false);
all.emplace_back("gauss_trunc",  true,
true,   false,
false);
all.emplace_back("sum_atan",  false,
true,   true,
false);
all.emplace_back("two_lorentz",  true,
false,   false,
false);
all.emplace_back("two_power",  true,
true,   false,
false);
all.emplace_back("two_power_gs",  true,
true,   false,
false);
all.emplace_back("akima_spline",  true,
false,   true,
true);
all.emplace_back("akima_spline_i",  false,
true,   false,
true);
all.emplace_back("akima_spline_ip",  false,
true,   false,
true);
all.emplace_back("cubic_spline",  true,
false,   true,
true);
all.emplace_back("cubic_spline_i",  false,
true,   false,
true);
all.emplace_back("cubic_spline_ip",  false,
true,   false,
true);
all.emplace_back("pedestal",  true,
true,   false,
false);
all.emplace_back("rational",  true,
true,   true,
false);
all.emplace_back("line_segment",  true,
false,   true,
true);
all.emplace_back("line_segment_i",  false,
true,   false,
true);
all.emplace_back("line_segment_ip",  false,
true,   false,
true);
all.emplace_back("nice_quadratic",  false,
false,   true,
false);
all.emplace_back("sum_cossq_s",  false,
true,   false,
false);
all.emplace_back("sum_cossq_sqrts",  false,
true,   false,
false);
all.emplace_back("sum_cossq_s_free",  false,
true,   false,
false);
return all;
}

}

ProfileParameterizationData::ProfileParameterizationData(
const std::string& name, bool allowedForPres, bool allowedForCurr,
bool allowedForIota, bool needsSplineData)
: name_(name),
needsSplineData_(needsSplineData),
allowedFor_({.pres = allowedForPres,
.curr = allowedForCurr,
.iota = allowedForIota}) {}

const std::string& ProfileParameterizationData::Name() const { return name_; }

bool ProfileParameterizationData::NeedsSplineData() const {
return needsSplineData_;
}

AllowedFor ProfileParameterizationData::IsAllowedFor() const {
return allowedFor_;
}

const std::vector<ProfileParameterizationData>& AllProfileParameterizations() {
static const std::vector<ProfileParameterizationData>* const kAll =
new std::vector<ProfileParameterizationData>(
BuildProfileParameterizations());
return *kAll;
}

const ProfileParameterizationData* FindProfileParameterization(
const std::string& name) {
for (const ProfileParameterizationData& entry :
AllProfileParameterizations()) {
if (entry.Name() == name) {
return &entry;
}
}
return nullptr;
}

bool IsProfileParameterizationAllowedFor(const std::string& name,
ProfileType type) {
const ProfileParameterizationData* const entry =
FindProfileParameterization(name);
if (entry == nullptr) {
return false;
}
const AllowedFor allowed = entry->IsAllowedFor();
switch (type) {
case ProfileType::PRESSURE:
return allowed.pres;
case ProfileType::CURRENT:
return allowed.curr;
case ProfileType::IOTA:
return allowed.iota;
}
return false;
}

}

// source: vmecpp/vmec/radial_partitioning/radial_partitioning.cc

#include <algorithm>
#include <iostream>

#include "absl/log/log.h"

namespace vmecpp {

RadialPartitioning::RadialPartitioning() {

adjustRadialPartitioning(1, 0, kNsDefault, false, false);
}

void RadialPartitioning::adjustRadialPartitioning(int num_threads,
int thread_id, int ns_input,
bool lfreeb, bool printout) {
ns_ = ns_input;
int num_surfaces_to_distribute = ns_ - 1;
if (lfreeb) {
num_surfaces_to_distribute = ns_;
}

if (num_threads > ns_ / 2) {
LOG(FATAL) << "Cannot make use of more than ns/2 (= " << (ns_ / 2)
<< ") threads, but tried to use " << num_threads << " threads.";
}

this->num_threads_ = num_threads;
this->thread_id_ = thread_id;

nsMinF1 = 0;
nsMaxF1 = ns_;

nsMinH = 0;
nsMaxH = ns_ - 1;

nsMinF = 0;
if (lfreeb) {
nsMaxF = ns_;
} else {
nsMaxF = ns_ - 1;
}

nsMinFi = 1;
nsMaxFi = ns_ - 1;

nsMaxFIncludingLcfs = nsMaxF;
if (nsMaxF1 == ns_) {
nsMaxFIncludingLcfs = ns_;
}

if (num_threads > 1) {
int work_per_cpu = num_surfaces_to_distribute / num_threads;
int work_remainder = num_surfaces_to_distribute % num_threads;

nsMinF = thread_id * work_per_cpu;
nsMaxF = (thread_id + 1) * work_per_cpu;
if (thread_id < work_remainder) {
nsMinF += thread_id;
nsMaxF += thread_id + 1;
} else {
nsMinF += work_remainder;
nsMaxF += work_remainder;
}

nsMinF1 = std::max(0, nsMinF - 1);
nsMaxF1 = std::min(ns_, nsMaxF + 1);

nsMinH = nsMinF1;
nsMaxH = nsMaxF1 - 1;

nsMinFi = std::max(1, nsMinF);
nsMaxFi = std::min(ns_ - 1, nsMaxF);

nsMaxFIncludingLcfs = nsMaxF;
if (nsMaxF1 == ns_) {
nsMaxFIncludingLcfs = ns_;
}
}

if (printout) {
std::cout << absl::StrFormat(
"thread %2d/%2d: "
"{nsMinF=%2d nsMaxF=%2d numFull=%2d} "
"{nsMinF1=%2d nsMaxF1=%2d numFull1=%2d} "
"{nsMinH=%2d nsMaxH=%2d numHalf=%2d} "
"{nsMinFi=%2d nsMaxFi=%2d numFullI=%2d}\n",
thread_id + 1, num_threads, nsMinF, nsMaxF, nsMaxF - nsMinF, nsMinF1,
nsMaxF1, nsMaxF1 - nsMinF1, nsMinH, nsMaxH, nsMaxH - nsMinH, nsMinFi,
nsMaxFi, nsMaxFi - nsMinFi);
}
}

int RadialPartitioning::get_num_threads() const { return num_threads_; }

int RadialPartitioning::get_thread_id() const { return thread_id_; }

bool RadialPartitioning::has_boundary() const { return nsMaxF1 == ns_; }

}

// source: vmecpp/vmec/radial_profiles/radial_profiles.cc

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"

using vmecpp::vmec_algorithm_constants::kEvenParity;
using vmecpp::vmec_algorithm_constants::kOddParity;

namespace {

constexpr std::array<double, 10> kGaussLegendreNodes01 = {
0.01304673574141414, 0.06746831665550774, 0.1602952158504878,
0.2833023029353764,  0.4255628305091844,  0.5744371694908156,
0.7166976970646236,  0.8397047841495122,  0.9325316833444923,
0.9869532642585859};
constexpr std::array<double, 10> kGaussLegendreWeights01 = {
0.03333567215434407, 0.0747256745752903, 0.1095431812579910,
0.1346333596549982,  0.1477621123573764, 0.1477621123573764,
0.1346333596549982,  0.1095431812579910, 0.0747256745752903,
0.03333567215434407};

template <typename Integrand>
constexpr double GaussLegendreIntegral01(Integrand integrand) {
double result = 0.0;
for (std::size_t i = 0; i < kGaussLegendreNodes01.size(); ++i) {
result += kGaussLegendreWeights01[i] * integrand(kGaussLegendreNodes01[i]);
}
return result;
}
static_assert(GaussLegendreIntegral01([](double) { return 2.0; }) == 2.0);
static_assert(GaussLegendreIntegral01([](double x) { return x; }) == 0.5);

double Coef(const Eigen::VectorXd& c, int i) {
return (i >= 0 && i < static_cast<int>(c.size())) ? c[i] : 0.0;
}

struct AkimaCoeffs {
int iv = 0;
std::vector<double> xloc;
std::vector<double> a, b, c, d;
};

AkimaCoeffs BuildAkimaCoeffs(const Eigen::VectorXd& xx,
const Eigen::VectorXd& yy) {
const int iv = static_cast<int>(xx.size());
const int sz = iv + 4;
AkimaCoeffs r;
r.iv = iv;
r.xloc.assign(sz, 0.0);
r.a.assign(sz, 0.0);
r.b.assign(sz, 0.0);
r.c.assign(sz, 0.0);
r.d.assign(sz, 0.0);
std::vector<double> yloc(sz, 0.0), m(sz, 0.0), t(sz, 0.0), dm(sz, 0.0),
p(sz, 0.0), q(sz, 0.0);

auto X = [&](int f) -> double& { return r.xloc[f + 1]; };
auto Y = [&](int f) -> double& { return yloc[f + 1]; };
auto M = [&](int f) -> double& { return m[f + 1]; };
auto DM = [&](int f) -> double& { return dm[f + 1]; };
auto P = [&](int f) -> double& { return p[f + 1]; };
auto Q = [&](int f) -> double& { return q[f + 1]; };
auto T = [&](int f) -> double& { return t[f + 1]; };
auto A = [&](int f) -> double& { return r.a[f + 1]; };
auto B = [&](int f) -> double& { return r.b[f + 1]; };
auto C = [&](int f) -> double& { return r.c[f + 1]; };
auto D = [&](int f) -> double& { return r.d[f + 1]; };

for (int i = 1; i <= iv; ++i) X(i) = xx[i - 1];
X(-1) = 2 * X(1) - X(3);
X(0) = X(1) + X(2) - X(3);
X(iv + 2) = 2 * X(iv) - X(iv - 2);
X(iv + 1) = X(iv) + X(iv - 1) - X(iv - 2);
for (int i = 1; i <= iv; ++i) Y(i) = yy[i - 1];

for (int f = -1; f <= iv + 1; ++f)
M(f) = (Y(f + 1) - Y(f)) / (X(f + 1) - X(f));

const double cl = (M(2) - M(1)) / (X(3) - X(1));
const double bl = M(1) - cl * (X(2) - X(1));

const double cr = (M(iv - 1) - M(iv - 2)) / (X(iv) - X(iv - 2));
const double br = M(iv - 1) - cr * (X(iv - 1) - X(iv));
Y(0) = Y(1) + bl * (X(0) - X(1)) + cl * std::pow(X(0) - X(1), 2);
Y(-1) = Y(1) + bl * (X(-1) - X(1)) + cl * std::pow(X(-1) - X(1), 2);
Y(iv + 1) =
Y(iv) + br * (X(iv + 1) - X(iv)) + cr * std::pow(X(iv + 1) - X(iv), 2);
Y(iv + 2) =
Y(iv) + br * (X(iv + 2) - X(iv)) + cr * std::pow(X(iv + 2) - X(iv), 2);

M(-1) = (Y(0) - Y(-1)) / (X(0) - X(-1));
M(0) = (Y(1) - Y(0)) / (X(1) - X(0));
M(iv) = (Y(iv + 1) - Y(iv)) / (X(iv + 1) - X(iv));
M(iv + 1) = (Y(iv + 2) - Y(iv + 1)) / (X(iv + 2) - X(iv + 1));

for (int f = -1; f <= iv; ++f) DM(f) = std::abs(M(f + 1) - M(f));
for (int i = 1; i <= iv; ++i) {
const double denom = DM(i) + DM(i - 2);
if (denom != 0.0) {
P(i) = DM(i) / denom;
Q(i) = DM(i - 2) / denom;
}
}
for (int i = 1; i <= iv; ++i) {
T(i) = P(i) * M(i - 1) + Q(i) * M(i);

if (P(i) + Q(i) < std::numeric_limits<double>::min())
T(i) = 0.5 * M(i - 1) + 0.5 * M(i);
}
for (int f = -1; f <= iv + 2; ++f) {
A(f) = Y(f);
B(f) = T(f);
}
for (int i = 1; i <= iv - 1; ++i) {
const double h = X(i + 1) - X(i);
C(i) = (3 * M(i) - T(i + 1) - 2 * T(i)) / h;
D(i) = (T(i + 1) + T(i) - 2 * M(i)) / (h * h);
}
return r;
}

Eigen::VectorXd BuildCubicSecondDerivatives(const Eigen::VectorXd& xa,
const Eigen::VectorXd& ya) {
const int n = static_cast<int>(xa.size());
double c =
((ya[2] - ya[0]) / (xa[2] - xa[0]) - (ya[1] - ya[0]) / (xa[1] - xa[0])) /
(xa[2] - xa[1]);
const double yp1 = (ya[1] - ya[0]) / (xa[1] - xa[0]) - c * (xa[1] - xa[0]);
c = ((ya[n - 3] - ya[n - 1]) / (xa[n - 3] - xa[n - 1]) -
(ya[n - 2] - ya[n - 1]) / (xa[n - 2] - xa[n - 1])) /
(xa[n - 3] - xa[n - 2]);
const double ypn = (ya[n - 2] - ya[n - 1]) / (xa[n - 2] - xa[n - 1]) -
c * (xa[n - 2] - xa[n - 1]);
Eigen::VectorXd y2 = Eigen::VectorXd::Zero(n);
Eigen::VectorXd u = Eigen::VectorXd::Zero(n);
y2[0] = -0.5;
u[0] = (3.0 / (xa[1] - xa[0])) * ((ya[1] - ya[0]) / (xa[1] - xa[0]) - yp1);
for (int i = 1; i <= n - 2; ++i) {
const double sig = (xa[i] - xa[i - 1]) / (xa[i + 1] - xa[i - 1]);
const double p = sig * y2[i - 1] + 2.0;
y2[i] = (sig - 1.0) / p;
u[i] = (6.0 *
((ya[i + 1] - ya[i]) / (xa[i + 1] - xa[i]) -
(ya[i] - ya[i - 1]) / (xa[i] - xa[i - 1])) /
(xa[i + 1] - xa[i - 1]) -
sig * u[i - 1]) /
p;
}
const double qn = 0.5;
const double un = (3.0 / (xa[n - 1] - xa[n - 2])) *
(ypn - (ya[n - 1] - ya[n - 2]) / (xa[n - 1] - xa[n - 2]));
y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
for (int k = n - 2; k >= 0; --k) y2[k] = y2[k] * y2[k + 1] + u[k];
return y2;
}

int CubicLowerIndex(const Eigen::VectorXd& xa, double x) {
int klo = 0;
int khi = static_cast<int>(xa.size()) - 1;
while (khi - klo > 1) {
const int k = (khi + klo) / 2;
if (xa[k] > x) {
khi = k;
} else {
klo = k;
}
}
return klo;
}
}

namespace vmecpp {

RadialProfiles::RadialProfiles(const RadialPartitioning* r,
HandoverStorage* m_h, const VmecINDATA* id,
const FlowControl* fc, int signOfJacobian,
double pDamp)
: r_(*r),
m_h_(*m_h),
id_(*id),
fc_(*fc),
signOfJacobian(signOfJacobian),
pDamp(pDamp) {
maxToroidalFlux = 0.0;
maxPoloidalFlux = 0.0;

currv = 0.0;
Itor = 0.0;
pressureScalingFactor = 0.0;

pmassType = ProfileParameterization::INVALID_PARAM;
piotaType = ProfileParameterization::INVALID_PARAM;
pcurrType = ProfileParameterization::INVALID_PARAM;

phipH.resize(r_.nsMaxH - r_.nsMinH);
chipH.resize(r_.nsMaxH - r_.nsMinH);
iotaH.resize(r_.nsMaxH - r_.nsMinH);
currH.resize(r_.nsMaxH - r_.nsMinH);
massH.resize(r_.nsMaxH - r_.nsMinH);
sqrtSH.resize(r_.nsMaxH - r_.nsMinH);

phipF.resize(r_.nsMaxF1 - r_.nsMinF1);
chipF.resize(r_.nsMaxF1 - r_.nsMinF1);
iotaF.resize(r_.nsMaxF1 - r_.nsMinF1);
sqrtSF.resize(r_.nsMaxF1 - r_.nsMinF1);
radialBlending.resize(r_.nsMaxF1 - r_.nsMinF1);

dVdsH.resize(r_.nsMaxH - r_.nsMinH);
presH.resize(r_.nsMaxH - r_.nsMinH);
bucoH.resize(r_.nsMaxH - r_.nsMinH);
bvcoH.resize(r_.nsMaxH - r_.nsMinH);

jcuruF.resize(r_.nsMaxFi - r_.nsMinFi);
jcurvF.resize(r_.nsMaxFi - r_.nsMinFi);
presgradF.resize(r_.nsMaxFi - r_.nsMinFi);
dVdsF.resize(r_.nsMaxFi - r_.nsMinFi);
equiF.resize(r_.nsMaxFi - r_.nsMinFi);

spectral_width.resize(r_.nsMaxF1 - r_.nsMinF1);
spectral_width.setZero();

scalxc.resize((r_.nsMaxF1 - r_.nsMinF1) * 2);

sm.resize(r_.nsMaxH - r_.nsMinH);
sm.setZero();
sp.resize(r_.nsMaxH - r_.nsMinH);
sp.setZero();
}

void RadialProfiles::setupInputProfiles() {
pmassType = findParameterization(id_.pmass_type, ProfileType::PRESSURE);
piotaType = findParameterization(id_.piota_type, ProfileType::IOTA);
pcurrType = findParameterization(id_.pcurr_type, ProfileType::CURRENT);

pressureScalingFactor = MU_0 * id_.pres_scale;

computeMagneticFluxes();
}

ProfileParameterization RadialProfiles::findParameterization(
const std::string& name, ProfileType intendedType) {
for (int i = 0; i < NUM_PARAM; ++i) {
if (name == AllProfileParameterizations()[i].Name()) {
bool isApplicable = false;
switch (intendedType) {
case ProfileType::PRESSURE:
isApplicable = AllProfileParameterizations()[i].IsAllowedFor().pres;
break;
case ProfileType::CURRENT:
isApplicable = AllProfileParameterizations()[i].IsAllowedFor().curr;
break;
case ProfileType::IOTA:
isApplicable = AllProfileParameterizations()[i].IsAllowedFor().iota;
break;
default:
std::cerr << absl::StrFormat("unknown profile: %s",
profileTypeToString(intendedType))
<< '\n';
break;
}

if (!isApplicable) {
std::cerr << absl::StrFormat(
"profile name '%s' is not applicable for %s profile",
AllProfileParameterizations()[i].Name(),
profileTypeToString(intendedType))
<< '\n';
return ProfileParameterization::INVALID_PARAM;
}

return static_cast<ProfileParameterization>(i);
}
}
return ProfileParameterization::INVALID_PARAM;
}

std::string RadialProfiles::profileTypeToString(ProfileType profileType) {
switch (profileType) {
case ProfileType::PRESSURE:
return "pressure";
case ProfileType::CURRENT:
return "current";
case ProfileType::IOTA:
return "iota";
default:
return "<unknown>";
}
}

void RadialProfiles::computeMagneticFluxes() {

const double phiedge = id_.phiedge * id_.bloat;
maxToroidalFlux = signOfJacobian * phiedge / (2.0 * M_PI);
double edgeToroidalFluxFromProfile = torflux(1.0);
if (edgeToroidalFluxFromProfile != 0.0) {
maxToroidalFlux /= edgeToroidalFluxFromProfile;
}

maxPoloidalFlux = maxToroidalFlux;
double edgePoloidalFluxFromProfile = polflux(1.0);
if (edgePoloidalFluxFromProfile != 0.0) {
maxPoloidalFlux /= edgePoloidalFluxFromProfile;
}
}

double RadialProfiles::torfluxDeriv(double x) {
double torflux_deriv = 0.0;
for (int i = static_cast<int>(id_.aphi.size()) - 1; i >= 0; i--) {
torflux_deriv = x * torflux_deriv + (i + 1) * id_.aphi[i];
}
return torflux_deriv;
}

double RadialProfiles::torflux(double x) {

double torflux = 0.0;
for (int i = static_cast<int>(id_.aphi.size()) - 1; i >= 0; i--) {
torflux = x * torflux + id_.aphi[i];
}
return torflux * x;
}

double RadialProfiles::polfluxDeriv(double x) {

double tf = std::min(torflux(x), 1.0);

double iota = evalIotaProfile(tf);

double polflux_deriv = iota * torfluxDeriv(x);
return polflux_deriv;
}

double RadialProfiles::polflux(double x) {

const int N = 100;
double delta_x = x / N;

double polflux = 0.0;
for (int i = 0; i <= N; ++i) {
double contribution = polfluxDeriv(i * delta_x);
if (i == 0 || i == N) {
polflux += 0.5 * contribution;
} else {
polflux += contribution;
}
}
polflux *= delta_x;

return polflux;
}

double RadialProfiles::evalMassProfile(double x) {

double normX = std::min(fabs(x * id_.bloat), 1.0);

double p = evalProfileFunction(pmassType, id_.am, id_.am_aux_s, id_.am_aux_f,
false, normX);

return p * pressureScalingFactor;
}

double RadialProfiles::evalIotaProfile(double x) {
double p = evalProfileFunction(piotaType, id_.ai, id_.ai_aux_s, id_.ai_aux_f,
false, x);

return p;
}

double RadialProfiles::evalCurrProfile(double x) {

double normX = std::min(std::abs(x * id_.bloat), 1.0);

double p = evalProfileFunction(pcurrType, id_.ac, id_.ac_aux_s, id_.ac_aux_f,
true, normX);

return p;
}

absl::Status RadialProfiles::CheckCurrentProfileEnclosesEdgeCurrent() {
if (id_.ncurr != 1) {
return absl::OkStatus();
}
const double edge_current = std::abs(evalCurrProfile(1.0));
double largest_current = edge_current;
static constexpr int kSamples = 100;
for (int i = 1; i < kSamples; ++i) {
largest_current =
std::max(largest_current,
std::abs(evalCurrProfile(static_cast<double>(i) / kSamples)));
}
if (largest_current > 0.0 && edge_current <= 1.0e-10 * largest_current) {
return absl::InvalidArgumentError(absl::StrFormat(
"the current profile '%s' encloses no net current at the boundary "
"(I(1) = %.3e against max |I| = %.3e), so it cannot be scaled to "
"curtor; give a profile with I(1) != 0 or prescribe iota with ncurr = "
"0",
id_.pcurr_type, edge_current, largest_current));
}
return absl::OkStatus();
}

double RadialProfiles::evalProfileFunction(const ProfileParameterization& param,
const Eigen::VectorXd& coeffs,
const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues,
bool shouldIntegrate, double normX) {
switch (param) {
case ProfileParameterization::POWER_SERIES:
return evalPowerSeries(coeffs, normX, shouldIntegrate);
case ProfileParameterization::POWER_SERIES_I:
return evalPowerSeriesI(coeffs, normX);
case ProfileParameterization::GAUSS_TRUNC:
return evalGaussTrunc(coeffs, normX, shouldIntegrate);
case ProfileParameterization::SUM_ATAN:
return evalSumAtan(coeffs, normX);
case ProfileParameterization::TWO_LORENTZ:
return evalTwoLorentz(coeffs, normX);
case ProfileParameterization::TWO_POWER:
return evalTwoPower(coeffs, normX, shouldIntegrate);
case ProfileParameterization::TWO_POWER_GS:
return evalTwoPowerGs(coeffs, normX, shouldIntegrate);
case ProfileParameterization::AKIMA_SPLINE:
case ProfileParameterization::AKIMA_SPLINE_I:
return evalAkima(splineKnots, splineValues, normX);
case ProfileParameterization::AKIMA_SPLINE_IP:
return evalAkimaIntegrated(splineKnots, splineValues, normX);
case ProfileParameterization::CUBIC_SPLINE:
case ProfileParameterization::CUBIC_SPLINE_I:
return evalCubic(splineKnots, splineValues, normX);
case ProfileParameterization::CUBIC_SPLINE_IP:
return evalCubicIntegrated(splineKnots, splineValues, normX);
case ProfileParameterization::PEDESTAL:
return evalPedestal(coeffs, normX, shouldIntegrate);
case ProfileParameterization::RATIONAL:
return evalRational(coeffs, normX);
case ProfileParameterization::LINE_SEGMENT:
case ProfileParameterization::LINE_SEGMENT_I:
return evalLineSegment(splineKnots, splineValues, normX);
case ProfileParameterization::LINE_SEGMENT_IP:
return evalLineSegmentIntegrated(splineKnots, splineValues, normX);
case ProfileParameterization::NICE_QUADRATIC:
return evalNiceQuadratic(coeffs, normX);
case ProfileParameterization::SUM_COSSQ_S:
case ProfileParameterization::SUM_COSSQ_SQRTS:
case ProfileParameterization::SUM_COSSQ_S_FREE:
default:
std::cerr
<< absl::StrFormat(
"profile parameterization '%s' not implemented yet",
AllProfileParameterizations()[static_cast<int>(param)].Name())
<< '\n';
}

return 0.0;
}

double RadialProfiles::evalPowerSeries(const Eigen::VectorXd& coeffs, double x,
bool should_integrate) {
double ret = 0.0;

for (Eigen::VectorXd::Index i = coeffs.size() - 1; i >= 0; i--) {
const double coeff = coeffs[i];
if (should_integrate) {
ret = x * ret + coeff / static_cast<double>(i + 1);
} else {
ret = x * ret + coeff;
}
}
if (should_integrate) {
ret *= x;
}
return ret;
}

double RadialProfiles::evalPowerSeriesI(const Eigen::VectorXd& coeffs,
double x) {
double ret = 0.0;
for (Eigen::VectorXd::Index i = coeffs.size() - 1; i >= 0; i--) {
const double coeff = coeffs[i];
ret = (ret + coeff) * x;
}
return ret;
}

double RadialProfiles::evalGaussTrunc(const Eigen::VectorXd& coeffs, double x,
bool shouldIntegrate) {

if (coeffs.size() < 2) {
LOG(WARNING)
<< "too few coefficients for 'gauss_trunc' profile; need 2, got "
<< coeffs.size() << "\n";
return 0.0;
}
const double a0 = coeffs[0];
const double a1 = coeffs[1];
const double edge = std::exp(-std::pow(1.0 / a1, 2));
if (!shouldIntegrate) {

return (a0 / (1.0 - edge)) * (std::exp(-std::pow(x / a1, 2)) - edge);
}

double integral = 0.0;
for (int i = 0; i < 10; ++i) {
const double xp = x * kGaussLegendreNodes01[i];
integral += kGaussLegendreWeights01[i] * a0 *
(std::exp(-std::pow(xp / a1, 2)) - edge);
}
return integral * x;
}

double RadialProfiles::evalSumAtan(const Eigen::VectorXd& coeffs, double x) {

double ret = 0.0;

if (coeffs.size() > 0) {
ret = coeffs[0];
}

if (x >= 1.0) {
if (coeffs.size() >= 2) {
ret += coeffs[1];
}
if (coeffs.size() >= 6) {
ret += coeffs[5];
}
if (coeffs.size() >= 10) {
ret += coeffs[9];
}
if (coeffs.size() >= 14) {
ret += coeffs[13];
}
if (coeffs.size() >= 18) {
ret += coeffs[17];
}
} else {
double atan_sum = 0.0;
if (coeffs.size() >= 5) {
atan_sum += coeffs[1] * std::atan(coeffs[2] * std::pow(x, coeffs[3]) /
std::pow(1 - x, coeffs[4]));
}
if (coeffs.size() >= 9) {
atan_sum += coeffs[5] * std::atan(coeffs[6] * std::pow(x, coeffs[7]) /
std::pow(1 - x, coeffs[8]));
}
if (coeffs.size() >= 13) {
atan_sum += coeffs[9] * std::atan(coeffs[10] * std::pow(x, coeffs[11]) /
std::pow(1 - x, coeffs[12]));
}
if (coeffs.size() >= 17) {
atan_sum += coeffs[13] * std::atan(coeffs[14] * std::pow(x, coeffs[15]) /
std::pow(1 - x, coeffs[16]));
}
if (coeffs.size() >= 21) {
atan_sum += coeffs[17] * std::atan(coeffs[18] * std::pow(x, coeffs[19]) /
std::pow(1 - x, coeffs[20]));
}
ret += 2.0 / M_PI * atan_sum;
}

return ret;
}

double RadialProfiles::evalTwoLorentz(const Eigen::VectorXd& coeffs, double x) {
if (coeffs.size() < 8) {
LOG(WARNING)
<< "too few coefficients for 'two_lorentz' profile; need 8, got "
<< coeffs.size() << "\n";
return 0.0;
}
double ret = 0.0;
ret =
coeffs[0] *
(coeffs[1] *
(1.0 /
std::pow(1.0 + std::pow(x / (coeffs[2] * coeffs[2]), coeffs[3]),
coeffs[4]) -
1.0 / std::pow(
1.0 + std::pow(1.0 / (coeffs[2] * coeffs[2]), coeffs[3]),
coeffs[4])) /
(1.0 - 1.0 / std::pow(1.0 + std::pow(1.0 / (coeffs[2] * coeffs[2]),
coeffs[3]),
coeffs[4])) +
(1.0 - coeffs[1]) *
(1.0 /
std::pow(1.0 + std::pow(x / (coeffs[5] * coeffs[5]), coeffs[6]),
coeffs[7]) -
1.0 / std::pow(
1.0 + std::pow(1.0 / (coeffs[5] * coeffs[5]), coeffs[6]),
coeffs[7])) /
(1.0 - 1.0 / std::pow(1.0 + std::pow(1.0 / (coeffs[5] * coeffs[5]),
coeffs[6]),
coeffs[7])));
return ret;
}

double RadialProfiles::evalTwoPower(const Eigen::VectorXd& coeffs, double x,
bool shouldIntegrate) {
if (coeffs.size() < 3) {
LOG(WARNING) << "too few coefficients for 'two_power' profile; need 3, got "
<< coeffs.size() << "\n";
return 0.0;
}
double ret = 0.0;
if (!shouldIntegrate) {

ret = coeffs[0] * std::pow(1.0 - std::pow(x, coeffs[1]), coeffs[2]);
} else {

int ngl = 10;
std::array glx = {0.01304673574141414, 0.06746831665550774,
0.1602952158504878,  0.2833023029353764,
0.4255628305091844,  0.5744371694908156,
0.7166976970646236,  0.8397047841495122,
0.9325316833444923,  0.9869532642585859};
std::array glw = {0.03333567215434407, 0.0747256745752903,
0.1095431812579910,  0.1346333596549982,
0.1477621123573764,  0.1477621123573764,
0.1346333596549982,  0.1095431812579910,
0.0747256745752903,  0.03333567215434407};

ret = 0.0;
for (int i = 0; i < ngl; ++i) {
const double xp = x * glx[i];
ret += glw[i] * coeffs[0] *
std::pow(1.0 - std::pow(xp, coeffs[1]), coeffs[2]);
}
ret *= x;
}
return ret;
}

double RadialProfiles::evalTwoPowerGs(const Eigen::VectorXd& coeffs, double x,
bool shouldIntegrate) {

if (coeffs.size() < 3) {
LOG(WARNING)
<< "too few coefficients for 'two_power_gs' profile; need 3, got "
<< coeffs.size() << "\n";
return 0.0;
}
auto raw = [&coeffs](double s) {
const double two_power =
Coef(coeffs, 0) *
std::pow(1.0 - std::pow(s, Coef(coeffs, 1)), Coef(coeffs, 2));
double gaussian = 1.0;
for (int i = 3; i <= 18; i += 3) {
const double amplitude = Coef(coeffs, i);
if (amplitude != 0.0) {
gaussian += amplitude *
std::exp(-std::pow(
(s - Coef(coeffs, i + 1)) / Coef(coeffs, i + 2), 2));
}
}
return two_power * gaussian;
};
if (!shouldIntegrate) {
return raw(x);
}

double integral = 0.0;
for (int i = 0; i < 10; ++i) {
integral += kGaussLegendreWeights01[i] * raw(x * kGaussLegendreNodes01[i]);
}
return integral * x;
}

double RadialProfiles::evalAkima(const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues,
double x) {

const int n = static_cast<int>(splineKnots.size());
if (n < 4 || n != static_cast<int>(splineValues.size())) {
LOG(WARNING)
<< "'akima_spline' profile needs at least 4 spline points; got " << n
<< " knots and " << splineValues.size() << " values\n";
return 0.0;
}
const AkimaCoeffs coeffs = BuildAkimaCoeffs(splineKnots, splineValues);
const int iv = coeffs.iv;
auto X = [&](int f) { return coeffs.xloc[f + 1]; };

if (x < X(1) || x > X(iv)) {
return 0.0;
}
if (x == X(iv)) {
return splineValues[iv - 1];
}
for (int i = 1; i <= iv - 1; ++i) {
if (x >= X(i) && x < X(i + 1)) {
const double dx = x - X(i);
return coeffs.a[i + 1] +
dx * (coeffs.b[i + 1] +
dx * (coeffs.c[i + 1] + coeffs.d[i + 1] * dx));
}
}
return 0.0;
}

double RadialProfiles::evalAkimaIntegrated(const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues,
double x) {

const int n = static_cast<int>(splineKnots.size());
if (n < 4 || n != static_cast<int>(splineValues.size())) {
LOG(WARNING)
<< "'akima_spline_ip' profile needs at least 4 spline points; got " << n
<< " knots and " << splineValues.size() << " values\n";
return 0.0;
}
const AkimaCoeffs coeffs = BuildAkimaCoeffs(splineKnots, splineValues);
const int iv = coeffs.iv;
auto X = [&](int f) { return coeffs.xloc[f + 1]; };
constexpr double kThird = 1.0 / 3.0;

std::vector<double> interval_integral(iv, 0.0);
for (int i = 1; i <= iv - 1; ++i) {
const double dx = X(i + 1) - X(i);
interval_integral[i] = coeffs.a[i + 1] * dx +
0.5 * coeffs.b[i + 1] * dx * dx +
kThird * coeffs.c[i + 1] * dx * dx * dx +
0.25 * coeffs.d[i + 1] * dx * dx * dx * dx;
}
if (x >= X(iv)) {
double y = 0.0;
for (int i = 1; i <= iv - 1; ++i) y += interval_integral[i];
return y;
}
if (x <= X(1)) {
return 0.0;
}
double y = 0.0;
for (int i = 1; i <= iv - 1; ++i) {
if (x >= X(i + 1)) {
y += interval_integral[i];
} else {
const double dx = x - X(i);
y += dx * (coeffs.a[i + 1] + dx * (0.5 * coeffs.b[i + 1] +
dx * (kThird * coeffs.c[i + 1] +
0.25 * coeffs.d[i + 1] * dx)));
return y;
}
}
return y;
}

double RadialProfiles::evalCubic(const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues,
double x) {

const int n = static_cast<int>(splineKnots.size());
if (n < 4 || n != static_cast<int>(splineValues.size())) {
LOG(WARNING)
<< "'cubic_spline' profile needs at least 4 spline points; got " << n
<< " knots and " << splineValues.size() << " values\n";
return 0.0;
}
if (x < splineKnots[0] || x > splineKnots[n - 1]) {
return 0.0;
}
const Eigen::VectorXd y2 =
BuildCubicSecondDerivatives(splineKnots, splineValues);
const int klo = CubicLowerIndex(splineKnots, x);
const int khi = klo + 1;
const double h = splineKnots[khi] - splineKnots[klo];
const double a = (splineKnots[khi] - x) / h;
const double b = (x - splineKnots[klo]) / h;
return a * splineValues[klo] + b * splineValues[khi] +
((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) /
6.0;
}

double RadialProfiles::evalCubicIntegrated(const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues,
double x) {

const int n = static_cast<int>(splineKnots.size());
if (n < 4 || n != static_cast<int>(splineValues.size())) {
LOG(WARNING)
<< "'cubic_spline_ip' profile needs at least 4 spline points; got " << n
<< " knots and " << splineValues.size() << " values\n";
return 0.0;
}
if (x < splineKnots[0] || x > splineKnots[n - 1]) {
return 0.0;
}
const Eigen::VectorXd y2 =
BuildCubicSecondDerivatives(splineKnots, splineValues);
const int klo = CubicLowerIndex(splineKnots, x);
const int khi = klo + 1;

double prefix = 0.0;
for (int i = 0; i < klo; ++i) {
const double dxa = splineKnots[i + 1] - splineKnots[i];
prefix += 0.5 * dxa * (splineValues[i + 1] + splineValues[i]) -
(dxa * dxa * dxa) * (y2[i + 1] + y2[i]) / 24.0;
}
const double h = splineKnots[khi] - splineKnots[klo];
const double dx = x - splineKnots[klo];
const double dya = splineValues[khi] - splineValues[klo];
const double dy2a = y2[khi] - y2[klo];
const double y = splineValues[klo] * dx + 0.5 * dya * dx * dx / h +
y2[klo] * dx * dx * (2 * dx - 3 * h) / 12.0 +
dy2a * dx * dx * (dx * dx - 2 * h * h) / (h * 24.0);
return y + prefix;
}

double RadialProfiles::evalPedestal(const Eigen::VectorXd& coeffs, double x,
bool shouldIntegrate) {

if (shouldIntegrate) {

double value = 0.0;
for (int i = 7; i >= 0; --i) {
value = x * value + Coef(coeffs, i) / (i + 1.0);
}
value *= x;

double width = Coef(coeffs, 11);
double amplitude = Coef(coeffs, 8);
double normalization;
if (width <= 0.0) {

amplitude = 0.0;
normalization = 0.0;
width = 1.0e30;
} else {
normalization = 1.0 / (std::tanh(2.0 * Coef(coeffs, 10) / width) -
std::tanh(2.0 * (Coef(coeffs, 10) - 1) / width));
}
const double width1 = std::max(Coef(coeffs, 16), 0.01);
const double width2 = std::max(Coef(coeffs, 20), 0.01);
const double g1 = (x - Coef(coeffs, 15)) / width1;
const double g3 = (-Coef(coeffs, 15)) / width1;
const double g2 = (x - Coef(coeffs, 19)) / width2;
const double g4 = (-Coef(coeffs, 19)) / width2;
value += normalization * amplitude *
(std::tanh(2.0 * Coef(coeffs, 10) / width) -
std::tanh(2.0 * (Coef(coeffs, 10) - std::sqrt(x)) / width)) +
Coef(coeffs, 13) * (std::tanh(g1) - std::tanh(g3)) +
Coef(coeffs, 17) * (std::tanh(g2) - std::tanh(g4));
return value;
}

double value = 0.0;
for (int i = 15; i >= 0; --i) {
value = x * value + Coef(coeffs, i);
}

double width = Coef(coeffs, 19);
double normalization;
if (width <= 0.0) {
normalization = 0.0;
width = 1.0e30;
} else {
normalization = 1.0 / (std::tanh(2.0 * Coef(coeffs, 18) / width) -
std::tanh(2.0 * (Coef(coeffs, 18) - 1) / width));
}
value += normalization * Coef(coeffs, 17) *
(std::tanh(2.0 * (Coef(coeffs, 18) - std::sqrt(x)) / width) -
std::tanh(2.0 * (Coef(coeffs, 18) - 1.0) / width));
return value;
}

double RadialProfiles::evalRational(const Eigen::VectorXd& coeffs, double x) {

double numerator = 0.0;
for (int i = 9; i >= 0; --i) {
numerator = x * numerator + Coef(coeffs, i);
}
double denominator = 0.0;
for (int i = static_cast<int>(coeffs.size()) - 1; i >= 10; --i) {
denominator = x * denominator + Coef(coeffs, i);
}
return (denominator != 0.0) ? numerator / denominator
: std::numeric_limits<double>::max();
}

double RadialProfiles::evalLineSegment(const Eigen::VectorXd& splineKnots,
const Eigen::VectorXd& splineValues,
double x) {
const int n = static_cast<int>(splineKnots.size());
if (n < 2 || n != static_cast<int>(splineValues.size())) {
return 0.0;
}
int ilow = 0;
if (x >= splineKnots[n - 1]) {
ilow = n - 2;
} else if (x > splineKnots[0]) {

const auto upper =
std::upper_bound(splineKnots.begin(), splineKnots.end(), x);
ilow = static_cast<int>(std::distance(splineKnots.begin(), upper)) - 1;
}
const double x0 = splineKnots[ilow];
const double x1 = splineKnots[ilow + 1];
const double y0 = splineValues[ilow];
const double y1 = splineValues[ilow + 1];
const double t = (x - x0) / (x1 - x0);
return (1.0 - t) * y0 + t * y1;
}

double RadialProfiles::evalLineSegmentIntegrated(
const Eigen::VectorXd& splineKnots, const Eigen::VectorXd& splineValues,
double x) {
const int n = static_cast<int>(splineKnots.size());
if (n < 2 || n != static_cast<int>(splineValues.size())) {
return 0.0;
}

const auto value_at = [this, &splineKnots, &splineValues](double xi) {
return evalLineSegment(splineKnots, splineValues, xi);
};

double result = 0.0;
double lower = 0.0;
for (int i = 0; i < n && splineKnots[i] < x; ++i) {
if (splineKnots[i] <= lower) {
continue;
}
const double upper = splineKnots[i];
result += 0.5 * (upper - lower) * (value_at(lower) + value_at(upper));
lower = upper;
}
result += 0.5 * (x - lower) * (value_at(lower) + value_at(x));

return result;
}

double RadialProfiles::evalNiceQuadratic(const Eigen::VectorXd& coeffs,
double x) {

return Coef(coeffs, 0) * (1.0 - x) + Coef(coeffs, 1) * x +
4.0 * Coef(coeffs, 2) * x * (1.0 - x);
}

void RadialProfiles::evalRadialProfiles(bool haveToFlipTheta,
VmecConstants& m_vmecconst) {

const double r00 = id_.rbc(0, id_.ntor);

currv = MU_0 * id_.curtor;

const double edgeCurrent = evalCurrProfile(1.0);
Itor = 0.0;

if (std::abs(edgeCurrent) > std::abs(DBL_EPSILON * id_.curtor)) {

Itor = signOfJacobian * currv / (2.0 * M_PI * edgeCurrent);
}

double local_rmsPhiP = 0.0;

for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
const double halfGridPos = (jH + 0.5) / (fc_.ns - 1.0);

sqrtSH[jH - r_.nsMinH] = std::sqrt(halfGridPos);

phipH[jH - r_.nsMinH] = maxToroidalFlux * torfluxDeriv(halfGridPos);
chipH[jH - r_.nsMinH] = maxToroidalFlux * polfluxDeriv(halfGridPos);

const double toroidalFlux = std::min(torflux(halfGridPos), 1.0);
iotaH[jH - r_.nsMinH] = evalIotaProfile(toroidalFlux);
currH[jH - r_.nsMinH] = evalCurrProfile(toroidalFlux);

if (haveToFlipTheta) {
chipH[jH - r_.nsMinH] *= -1.0;
iotaH[jH - r_.nsMinH] *= -1.0;
}

currH[jH - r_.nsMinH] *= Itor;

const double vpnorm = maxToroidalFlux * torfluxDeriv(halfGridPos);
const double massEvalPos = std::min(halfGridPos, id_.spres_ped);

const double massEvalTorFlux = std::min(torflux(massEvalPos), 1.0);
const double mass = evalMassProfile(massEvalTorFlux);
massH[jH - r_.nsMinH] = mass * std::pow(std::abs(vpnorm) * r00, id_.gamma);

if (jH < r_.nsMaxH - 1 || jH == fc_.ns - 2) {
local_rmsPhiP += phipH[jH - r_.nsMinH] * phipH[jH - r_.nsMinH];
}
}

m_vmecconst.rmsPhiP += local_rmsPhiP;

const double sqrtS1 = sqrt(1.0 / (fc_.ns - 1));

for (int jF1 = r_.nsMinF1; jF1 < r_.nsMaxF1; ++jF1) {
const double fullGridPos = jF1 / (fc_.ns - 1.0);

sqrtSF[jF1 - r_.nsMinF1] = std::sqrt(fullGridPos);

phipF[jF1 - r_.nsMinF1] = maxToroidalFlux * torfluxDeriv(fullGridPos);
chipF[jF1 - r_.nsMinF1] = maxToroidalFlux * polfluxDeriv(fullGridPos);

const double toroidalFlux = std::min(torflux(fullGridPos), 1.0);
iotaF[jF1 - r_.nsMinF1] = evalIotaProfile(toroidalFlux);

radialBlending[jF1 - r_.nsMinF1] = 2.0 * pDamp * (1.0 - fullGridPos);

scalxc[(jF1 - r_.nsMinF1) * 2 + kEvenParity] = 1.0;

scalxc[(jF1 - r_.nsMinF1) * 2 + kOddParity] =
1.0 / std::max(sqrtSF[jF1 - r_.nsMinF1], sqrtS1);
}

if (r_.nsMaxF1 == fc_.ns) {
sqrtSF[fc_.ns - 1 - r_.nsMinF1] = 1.0;
}

for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
const int jFi = jH;
const int jFo = jH + 1;
sm[jH - r_.nsMinH] = sqrtSH[jH - r_.nsMinH] / sqrtSF[jFo - r_.nsMinF1];
if (jH > 0) {
sp[jH - r_.nsMinH] = sqrtSH[jH - r_.nsMinH] / sqrtSF[jFi - r_.nsMinF1];
}
}
if (r_.nsMinH == 0) {

sp[0] = sm[0];
}
}

void RadialProfiles::AccumulateVolumeAveragedSpectralWidth() const {
SpectralWidthContribution spectral_width_contribution = {.numerator = 0.0,
.denominator = 0.0};

for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {

if (jH < r_.nsMaxH - 1 || jH == fc_.ns - 2) {
const int jFi = jH;
const int jFo = jH + 1;

const double spectral_width_on_half_grid =
(spectral_width[jFo - r_.nsMinF1] +
spectral_width[jFi - r_.nsMinF1]) /
2.0;
spectral_width_contribution.numerator +=
spectral_width_on_half_grid * dVdsH[jH - r_.nsMinH];

spectral_width_contribution.denominator += dVdsH[jH - r_.nsMinH];
}
}

SumOverThreads(&spectral_width_contribution.numerator, 1, r_.get_thread_id(),
r_.get_num_threads(), m_h_.thread_reduce_slots.data(),
m_h_.SpectralWidthNumerator());
SumOverThreads(&spectral_width_contribution.denominator, 1,
r_.get_thread_id(), r_.get_num_threads(),
m_h_.thread_reduce_slots.data(),
m_h_.SpectralWidthDenominator());
}

}

// source: vmecpp/vmec/thread_local_storage/thread_local_storage.cc

namespace vmecpp {

ThreadLocalStorage::ThreadLocalStorage(const Sizes* s) : s_(*s) {
const int nZnT = s_.nZnT;
const int nZnT3d = s_.lthreed ? nZnT : 0;

r1e_i.setZero(nZnT);
r1o_i.setZero(nZnT);
rue_i.setZero(nZnT);
ruo_i.setZero(nZnT);
rve_i.setZero(nZnT3d);
rvo_i.setZero(nZnT3d);
z1e_i.setZero(nZnT);
z1o_i.setZero(nZnT);
zue_i.setZero(nZnT);
zuo_i.setZero(nZnT);
zve_i.setZero(nZnT3d);
zvo_i.setZero(nZnT3d);
lue_i.setZero(nZnT);
luo_i.setZero(nZnT);
lve_i.setZero(nZnT3d);
lvo_i.setZero(nZnT3d);
bsubu_i.setZero(nZnT);
bsubv_i.setZero(nZnT);
gvv_gsqrt_i.setZero(nZnT);
guv_bsupu_i.setZero(nZnT);
P_i.setZero(nZnT);
rup_i.setZero(nZnT);
zup_i.setZero(nZnT);
rsp_i.setZero(nZnT);
zsp_i.setZero(nZnT);
taup_i.setZero(nZnT);
gbubu_i.setZero(nZnT);
gbubv_i.setZero(nZnT);
gbvbv_i.setZero(nZnT);
P_o.setZero(nZnT);
rup_o.setZero(nZnT);
zup_o.setZero(nZnT);
rsp_o.setZero(nZnT);
zsp_o.setZero(nZnT);
taup_o.setZero(nZnT);
gbubu_o.setZero(nZnT);
gbubv_o.setZero(nZnT);
gbvbv_o.setZero(nZnT);
P_avg.setZero(nZnT);
P_wavg.setZero(nZnT);
gbubu_avg.setZero(nZnT);
gbubu_wavg.setZero(nZnT);
gbvbv_avg.setZero(nZnT);
gbvbv_wavg.setZero(nZnT);
gbubv_avg.setZero(nZnT);
gbubv_wavg.setZero(nZnT);
}
}

// source: vmecpp/vmec/vmec/vmec.cc

// header: vmecpp/vmec/vmec/vmec.h

#ifndef VMECPP_VMEC_VMEC_VMEC_H_
#define VMECPP_VMEC_VMEC_VMEC_H_

#include <climits>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace vmecpp {

enum class MultigridInterpolationScheme : std::uint8_t {

kLinear,

kCubic,

kCubicRho,
};

struct HotRestartState {
WOutFileContents wout;
VmecINDATA indata;

HotRestartState(WOutFileContents wout, VmecINDATA indata)
: wout(std::move(wout)), indata(std::move(indata)) {}

explicit HotRestartState(const OutputQuantities& output_quantities)
: wout(output_quantities.wout), indata(output_quantities.indata) {}

explicit HotRestartState(OutputQuantities&& output_quantities)
: wout(std::move(output_quantities.wout)),
indata(std::move(output_quantities.indata)) {}
};

using InterruptCallback = std::function<bool()>;

absl::StatusOr<OutputQuantities> run(
const VmecINDATA& indata,
std::optional<HotRestartState> initial_state = std::nullopt,
std::optional<int> max_threads = std::nullopt,
OutputMode verbose = OutputMode::kLegacy,
InterruptCallback interrupt_callback = nullptr);

absl::StatusOr<OutputQuantities> run(
const VmecINDATA& indata,
const makegrid::MagneticFieldResponseTable& magnetic_response_table,
std::optional<HotRestartState> initial_state = std::nullopt,
std::optional<int> max_threads = std::nullopt,
OutputMode verbose = OutputMode::kLegacy,
InterruptCallback interrupt_callback = nullptr);

class Vmec {
public:

explicit Vmec(const VmecINDATA& indata,
std::optional<int> max_threads = std::nullopt,
OutputMode verbose = OutputMode::kLegacy,
InterruptCallback interrupt_callback = nullptr);

Vmec(const Vmec&) = delete;
Vmec& operator=(const Vmec&) = delete;
Vmec(Vmec&&) = delete;
Vmec& operator=(Vmec&&) = delete;

static constexpr int kSignOfJacobian = -1;

static constexpr double kPDamp = 0.05;

static absl::StatusOr<std::unique_ptr<Vmec>> FromIndata(
const VmecINDATA& indata,
const makegrid::MagneticFieldResponseTable* magnetic_response_table =
nullptr,
std::optional<int> max_threads = std::nullopt,
OutputMode verbose = OutputMode::kLegacy,
InterruptCallback interrupt_callback = nullptr);

absl::StatusOr<bool> run(
const VmecCheckpoint& checkpoint = VmecCheckpoint::NONE,
int iterations_before_checkpointing = INT_MAX,
int maximum_multi_grid_step = 500,
std::optional<HotRestartState> initial_state = std::nullopt);

void SetupVacuumSolvers();

absl::StatusOr<bool> InitializeRadial(
VmecCheckpoint checkpoint, int maximum_iterations, int nsval, int ns_old,
double& m_delt0,
const std::optional<HotRestartState>& initial_state = std::nullopt,
std::optional<MultigridInterpolationScheme> interpolation_scheme =
std::nullopt);
absl::StatusOr<bool> SolveEquilibrium(VmecCheckpoint checkpoint,
int maximum_iterations);
void RestartIteration(double& m_delt0r, int thread_id);
absl::StatusOr<bool> Evolve(VmecCheckpoint checkpoint, int maximum_iterations,
double time_step, int thread_id,
bool& m_liter_flag);
void Printout(double delt0r, int thread_id, int iter2);
absl::StatusOr<bool> UpdateForwardModel(VmecCheckpoint checkpoint,
int maximum_iterations,
int thread_id);
void PerformTimeStep(double fac, double b1, double time_step, int thread_id);
void InterpolateToNextMultigridStep(
int ns_new, int ns_old,
const std::vector<std::unique_ptr<RadialProfiles>>& p,
const std::vector<std::unique_ptr<RadialPartitioning>>& r_new,
const std::vector<std::unique_ptr<RadialPartitioning>>& r_old,
std::vector<std::unique_ptr<FourierGeometry>>& m_x_new,
std::vector<std::unique_ptr<FourierGeometry>>& m_x_old,
std::optional<MultigridInterpolationScheme> interpolation_scheme =
std::nullopt);

bool updateFwdModel(IdealMhdModel& m_m, FourierGeometry& m_decomposed_x,
FourierGeometry& m_physical_x, HandoverStorage& m_h,
FourierForces& m_decomposed_f,
FourierForces& m_physical_f, const RadialPartitioning& r,
FlowControl& m_fc, int thread_id,
const VmecCheckpoint& checkpoint = VmecCheckpoint::NONE,
int maximum_iterations = INT_MAX);

void evolve(const RadialPartitioning& r, FourierGeometry& m_decomposed_x,
FourierVelocity& m_decomposed_v,
const FourierForces& decomposed_f, const FlowControl& fc);

void performTimeStep(const Sizes& s, const FlowControl& fc,
const RadialPartitioning& r, double velocityScale,
double conjugationParameter, double time_step,
FourierGeometry& m_decomposed_x,
FourierVelocity& m_decomposed_v,
const FourierForces& decomposed_f,
HandoverStorage& m_h_) const;

int get_ivac() const { return static_cast<int>(vacuum_pressure_state_); }
int get_num_eqsolve_retries() const { return num_eqsolve_retries_; }
VmecStatus get_status() const { return status_; }
int get_iter1() const { return iter1_; }
int get_iter2() const { return iter2_; }
int get_last_preconditioner_update() const {
return last_preconditioner_update_;
}
int get_last_full_update_nestor() const { return last_full_update_nestor_; }
int get_jacob_off() { return jacob_off_; }

VmecINDATA indata_;
Sizes s_;
FourierBasisFastPoloidal t_;
Boundaries b_;
VmecConstants constants_;
HandoverStorage h_;
FlowControl fc_;
MGridProvider mgrid_;
OutputQuantities output_quantities_;

int num_threads_;

int vac_num_threads_ = 0;
std::vector<std::unique_ptr<RadialPartitioning>> r_;
std::vector<std::unique_ptr<ThreadLocalStorage>> ls_;
std::vector<std::unique_ptr<RadialProfiles>> p_;
std::vector<std::unique_ptr<FreeBoundaryBase>> fb_vac_;
std::vector<std::unique_ptr<TangentialPartitioning>> tp_vac_;
std::vector<std::unique_ptr<IdealMhdModel>> m_;
std::vector<std::unique_ptr<FourierGeometry>> decomposed_x_;
std::vector<std::unique_ptr<FourierGeometry>> physical_x_backup_;
std::vector<std::unique_ptr<FourierGeometry>> physical_x_;
std::vector<std::unique_ptr<FourierForces>> decomposed_f_;
std::vector<std::unique_ptr<FourierForces>> physical_f_;
std::vector<std::unique_ptr<FourierVelocity>> decomposed_v_;

std::vector<std::unique_ptr<FourierGeometry>> old_xc_scaled_;
std::vector<std::unique_ptr<RadialPartitioning>> old_r_;

Eigen::VectorXd matrixShare;

Eigen::PartialPivLU<Eigen::MatrixXd> lu_decomposition;
Eigen::VectorXd bvecShare;

Eigen::VectorXd vacuum_reduce_slots_;

private:
enum class SolveEqLoopStatus : std::uint8_t {
NORMAL_TERMINATION,
CHECKPOINT_REACHED,
MUST_RETRY
};

absl::StatusOr<SolveEqLoopStatus> SolveEquilibriumLoop(
int thread_id, int maximum_iterations, VmecCheckpoint checkpoint,
bool& m_lreset_internal, bool& m_liter_flag);

bool verbose_;

IterationLogger logger_;

InterruptCallback interrupt_callback_;

bool interrupted_ = false;

VacuumPressureState vacuum_pressure_state_;

int jacob_off_ = 0;

int num_eqsolve_retries_;

VmecStatus status_;

int iter2_;

int iter1_;

static constexpr int kNDamp = 10;

Eigen::VectorXd invTau_;

int last_preconditioner_update_;

int last_full_update_nestor_;
};

}

#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

namespace {

void UpdateStatusForThread(absl::Status& m_status_of_all_threads, int thread_id,
const absl::Status& thread_status) {
CHECK(!thread_status.ok()) << "UpdateStatusForThread expects an error status";

const auto thread_msg =
absl::StrFormat("Thread %i:\n\t%s\n", thread_id, thread_status.message());

if (m_status_of_all_threads.ok()) {
m_status_of_all_threads = absl::InternalError(absl::StrCat(
"There was an error in one or more threads during a VMEC++ run:\n",
thread_msg));
return;
}

m_status_of_all_threads = absl::InternalError(
absl::StrCat(m_status_of_all_threads.message(), thread_msg));
}

absl::Status CheckInitialState(const vmecpp::HotRestartState& initial_state,
const vmecpp::VmecINDATA& indata) {
const auto msg_start = "Mismatch in variable '";
const auto msg_end =
"' between hot restart initial state and indata. This is not supported "
"yet.";

if (initial_state.indata.lasym != indata.lasym) {
return absl::InvalidArgumentError(
absl::StrCat(msg_start, "lasym", msg_end));
}

if (initial_state.indata.mpol != indata.mpol) {
return absl::InvalidArgumentError(absl::StrCat(msg_start, "mpol", msg_end));
}
if (initial_state.indata.ntor != indata.ntor) {
return absl::InvalidArgumentError(absl::StrCat(msg_start, "ntor", msg_end));
}

if (initial_state.wout.ns != indata.ns_array[0]) {
return absl::InvalidArgumentError(absl::StrFormat(
"%sns_array%s The wout of the initial state has ns = %d, but "
"ns_array[0] = %d.",
msg_start, msg_end, initial_state.wout.ns, indata.ns_array[0]));
}

return absl::OkStatus();
}
}

absl::StatusOr<vmecpp::OutputQuantities> vmecpp::run(
const VmecINDATA& indata, std::optional<HotRestartState> initial_state,
std::optional<int> max_threads, OutputMode verbose,
InterruptCallback interrupt_callback) {
auto maybe_vmec = Vmec::FromIndata(indata, nullptr, max_threads, verbose,
std::move(interrupt_callback));
if (!maybe_vmec.ok()) {
return maybe_vmec.status();
}
Vmec& v = **maybe_vmec;

absl::StatusOr<bool> s =
v.run(VmecCheckpoint::NONE, INT_MAX, 500, std::move(initial_state));

if (!s.ok()) {
return s.status();
}

return std::move(v.output_quantities_);
}

absl::StatusOr<vmecpp::OutputQuantities> vmecpp::run(
const VmecINDATA& indata,
const makegrid::MagneticFieldResponseTable& magnetic_response_table,
std::optional<HotRestartState> initial_state,
std::optional<int> max_threads, OutputMode verbose,
InterruptCallback interrupt_callback) {
auto maybe_vmec =
Vmec::FromIndata(indata, &magnetic_response_table, max_threads, verbose,
std::move(interrupt_callback));
if (!maybe_vmec.ok()) {
return maybe_vmec.status();
}
Vmec& v = **maybe_vmec;

absl::StatusOr<bool> s =
v.run(VmecCheckpoint::NONE, INT_MAX, 500, std::move(initial_state));

if (!s.ok()) {
return s.status();
}

return std::move(v.output_quantities_);
}

namespace vmecpp {

absl::StatusOr<std::unique_ptr<Vmec>> Vmec::FromIndata(
const VmecINDATA& indata,
const makegrid::MagneticFieldResponseTable* magnetic_response_table,
std::optional<int> max_threads, OutputMode verbose,
InterruptCallback interrupt_callback) {

absl::Status is_indata_consistent =
IsConsistent(indata,  false);
if (!is_indata_consistent.ok()) {
return is_indata_consistent;
}

auto v = std::make_unique<Vmec>(indata, max_threads, verbose,
std::move(interrupt_callback));

if (indata.lfreeb) {
absl::Status s{};
if (magnetic_response_table == nullptr) {
s = v->mgrid_.LoadFile(indata.mgrid_file, indata.extcur);
} else {
s = v->mgrid_.LoadFields(*magnetic_response_table, indata.extcur);
}
if (!s.ok()) {
return s;
}
}

return v;
}

Vmec::Vmec(const VmecINDATA& indata, std::optional<int> max_threads,
OutputMode verbose, InterruptCallback interrupt_callback)
: indata_(indata),
s_(indata_),
t_(&s_),
b_(&s_, &t_, indata_.signgs),
h_(&s_),
fc_(indata_.lfreeb, indata_.delt,
static_cast<int>(indata_.ns_array.size()), max_threads),
verbose_(verbose != OutputMode::kSilent),
logger_(std::cout, verbose),
interrupt_callback_(std::move(interrupt_callback)),
vacuum_pressure_state_(VacuumPressureState::kOff),
status_(VmecStatus::NORMAL_TERMINATION),
iter2_(1),
iter1_(iter2_),
invTau_(kNDamp),
last_preconditioner_update_(0),
last_full_update_nestor_(0) {

fc_.haveToFlipTheta = b_.setupFromIndata(indata_, verbose_);

if (fc_.lfreeb) {

int nf = s_.ntor;

int mf = s_.mpol + 1;
int mnpd = (2 * nf + 1) * (mf + 1);

int mnpd_dim = s_.lasym ? 2 * mnpd : mnpd;
matrixShare.setZero(mnpd_dim * mnpd_dim);
bvecShare.setZero(mnpd_dim);

h_.vacuum_magnetic_pressure.setZero(s_.nZnT);
h_.initial_plasma_pressure_at_boundary.setZero(s_.nZnT);
h_.initial_vacuum_pressure_at_boundary.setZero(s_.nZnT);
h_.edge_total_pressure.setZero(s_.nZnT);
h_.vacuum_b_r.setZero(s_.nZnT);
h_.vacuum_b_phi.setZero(s_.nZnT);
h_.vacuum_b_z.setZero(s_.nZnT);
}
}

absl::StatusOr<bool> Vmec::run(const VmecCheckpoint& checkpoint,
const int iterations_before_checkpointing,
const int maximum_multi_grid_step,
std::optional<HotRestartState> initial_state) {
if (maximum_multi_grid_step < 1) {
return absl::InvalidArgumentError(
absl::StrFormat("maximum_multi_grid_step must be at least 1, but is %d",
maximum_multi_grid_step));
}

if (indata_.lfreeb) {
if (!mgrid_.IsLoaded()) {

absl::Status status = mgrid_.LoadFile(indata_.mgrid_file, indata_.extcur);
if (!status.ok()) {
return status;
}
}

if (mgrid_.numPhi != s_.nZeta) {
const std::string raised_from =
s_.nZeta == indata_.nzeta
? ""
: absl::StrFormat(
" (nzeta = %d in VmecINDATA, raised to the "
"minimum for ntor = %d)",
indata_.nzeta, indata_.ntor);
return absl::InvalidArgumentError(absl::StrFormat(
"MGridProvider has %d phi grid points, but the run has %d toroidal "
"grid points%s. Please ensure that the two are consistent.",
mgrid_.numPhi, s_.nZeta, raised_from));
}
if (mgrid_.nfp != indata_.nfp) {
return absl::InvalidArgumentError(absl::StrFormat(
"MGridProvider has %d field periods, but VmecINDATA has nfp = %d. "
"Please ensure that the two are consistent.",
mgrid_.nfp, indata_.nfp));
}
}

auto is_indata_consistent =
IsConsistent(indata_,  verbose_);
if (!is_indata_consistent.ok()) {
return is_indata_consistent;
}

if (initial_state.has_value()) {
absl::Status status = CheckInitialState(*initial_state, indata_);
if (!status.ok()) {
return status;
}
}

num_eqsolve_retries_ = 0;

fc_.ns_old = 0;
fc_.delt0r = indata_.delt;

bool giving_up = false;

for (jacob_off_ = 0; jacob_off_ < 2; ++jacob_off_) {

if (fc_.lfreeb && initial_state.has_value()) {
vacuum_pressure_state_ = VacuumPressureState::kInitialized;
}

fc_.ns_min = 3;

const int max_grids = std::min(fc_.multi_ns_grid, maximum_multi_grid_step);
for (int igrid = -jacob_off_; igrid < max_grids; igrid++) {
constants_.reset();

if (igrid < 0) {

fc_.nsval = 3;
fc_.ftolv = 1.0e-4;

vacuum_pressure_state_ = VacuumPressureState::kOff;
} else {

fc_.nsval = indata_.ns_array[igrid];
if (fc_.nsval < fc_.ns_min) {

continue;
}

fc_.ns_min = fc_.nsval;

fc_.ftolv = indata_.ftol_array[igrid];
fc_.niterv = indata_.niter_array[igrid];
}

{
const size_t cap =
fc_.force_residual_r.size() + static_cast<size_t>(fc_.niterv) + 1;
fc_.force_residual_r.reserve(cap);
fc_.force_residual_z.reserve(cap);
fc_.force_residual_lambda.reserve(cap);
fc_.mhd_energy.reserve(cap);
fc_.delbsq.reserve(cap);
fc_.restart_reasons.reserve(cap);
}

logger_.BeginStage(igrid, max_grids + jacob_off_, fc_.nsval, s_.mnmax,
fc_.ftolv, fc_.niterv, fc_.lfreeb);

const absl::StatusOr<bool> initialized =
InitializeRadial(checkpoint, iterations_before_checkpointing,
fc_.nsval, fc_.ns_old, fc_.delt0r, initial_state);
if (!initialized.ok()) {
return initialized.status();
}
if (*initialized) {
return true;
}

const absl::StatusOr<bool> reached_checkpoint =
SolveEquilibrium(checkpoint, iterations_before_checkpointing);
if (!reached_checkpoint.ok() || *reached_checkpoint == true) {
return reached_checkpoint;
}

if (status_ != VmecStatus::NORMAL_TERMINATION &&
status_ != VmecStatus::SUCCESSFUL_TERMINATION) {
if (!indata_.return_outputs_even_if_not_converged) {
const auto msg = absl::StrFormat(
"FATAL ERROR in SolveEquilibrium: %s\n"
"VmecINDATA had these contents:\n%s",
VmecStatusAsString(status_), *indata_.ToJson());

return absl::InternalError(msg);
}

if (verbose_) {
std::cout << absl::StrFormat(
"WARNING: FATAL ERROR in SolveEquilibrium: %s\n"
"return_outputs_even_if_not_converged is set, so returning "
"best-effort (likely unphysical) output for debugging "
"purposes.\n",
VmecStatusAsString(status_));
}
giving_up = true;
break;
}

}

if (giving_up) {
break;
}

if (status_ != VmecStatus::BAD_JACOBIAN) {

break;
}

}

if (status_ != VmecStatus::SUCCESSFUL_TERMINATION &&
!indata_.return_outputs_even_if_not_converged) {

const auto msg = absl::StrFormat(
"VMEC++ did not converge: %s. Completed %d/%d iterations at ns = "
"%d without meeting ftol = %.3e; final force residuals were "
"fsqr = %.3e, fsqz = %.3e, fsql = %.3e.",
VmecStatusAsString(status_), iter2_ - 1, fc_.niterv, fc_.nsval,
fc_.ftolv, fc_.fsqr, fc_.fsqz, fc_.fsql);
return absl::InternalError(msg);
}

if (!h_.vacuum_status.ok()) {
if (!indata_.return_outputs_even_if_not_converged) {
return h_.vacuum_status;
}
status_ = VmecStatus::UNRECOVERABLE_ERROR;
}

if (fc_.lfreeb) {
h_.vacuum_potential = bvecShare;
}

output_quantities_ = vmecpp::ComputeOutputQuantities(
indata_.signgs, indata_, s_, fc_, constants_, t_, h_, mgrid_.mgrid_mode,
mgrid_.coil_group_names, r_, decomposed_x_, m_, p_, checkpoint,
vacuum_pressure_state_, status_, iter2_);

{
const auto& w = output_quantities_.wout;
RunSummary summary;
summary.converged = (status_ == VmecStatus::SUCCESSFUL_TERMINATION);
summary.total_iterations = w.itfsq;
summary.num_jacobian_resets = fc_.ijacob;
summary.fsqr = w.fsqr;
summary.fsqz = w.fsqz;
summary.fsql = w.fsql;
summary.ftolv = fc_.ftolv;
summary.betatot = w.betatotal;
summary.betapol = w.betapol;
summary.betator = w.betator;
summary.w_mhd = h_.mhdEnergy * 4.0 * M_PI * M_PI;
summary.rax = w.Rmajor_p;
summary.aminor = w.Aminor_p;
summary.rmajor = w.Rmajor_p;
summary.b0 = w.b0;
logger_.EndRun(summary);
}

return false;
}

void Vmec::SetupVacuumSolvers() {

vac_num_threads_ = vmec_adjust_vacuum_num_threads(fc_.max_threads(), s_.nZnT);

#ifdef _OPENMP

omp_set_nested(1);
omp_set_max_active_levels(2);
#endif

vacuum_reduce_slots_.setZero(static_cast<Eigen::Index>(vac_num_threads_) *
matrixShare.size());

fb_vac_.resize(vac_num_threads_);
tp_vac_.resize(vac_num_threads_);

for (int vac_thread_id = 0; vac_thread_id < vac_num_threads_;
++vac_thread_id) {
tp_vac_[vac_thread_id] = std::make_unique<TangentialPartitioning>(
s_.nZnT, vac_num_threads_, vac_thread_id);

if (indata_.free_boundary_method == FreeBoundaryMethod::NESTOR) {
fb_vac_[vac_thread_id] = std::make_unique<Nestor>(
&s_, tp_vac_[vac_thread_id].get(), &mgrid_,
std::span<double>(matrixShare.data(), matrixShare.size()),
std::span<double>(bvecShare.data(), bvecShare.size()),
std::span<double>(h_.vacuum_magnetic_pressure.data(),
h_.vacuum_magnetic_pressure.size()),
&lu_decomposition,
std::span<double>(h_.vacuum_b_r.data(), h_.vacuum_b_r.size()),
std::span<double>(h_.vacuum_b_phi.data(), h_.vacuum_b_phi.size()),
std::span<double>(h_.vacuum_b_z.data(), h_.vacuum_b_z.size()),
std::span<double>(vacuum_reduce_slots_.data(),
vacuum_reduce_slots_.size()));
} else if (indata_.free_boundary_method == FreeBoundaryMethod::ONLY_COILS) {
fb_vac_[vac_thread_id] = std::make_unique<OnlyCoils>(
&s_, tp_vac_[vac_thread_id].get(), &mgrid_,
std::span<double>(h_.vacuum_magnetic_pressure.data(),
h_.vacuum_magnetic_pressure.size()),
std::span<double>(h_.vacuum_b_r.data(), h_.vacuum_b_r.size()),
std::span<double>(h_.vacuum_b_phi.data(), h_.vacuum_b_phi.size()),
std::span<double>(h_.vacuum_b_z.data(), h_.vacuum_b_z.size()),
std::span<double>(vacuum_reduce_slots_.data(),
vacuum_reduce_slots_.size()));
} else {
LOG(FATAL) << absl::StrCat("free boundary method '",
ToString(indata_.free_boundary_method),
"' not implemented yet");
}
}
}

absl::StatusOr<bool> Vmec::InitializeRadial(
VmecCheckpoint checkpoint, int iterations_before_checkpointing, int nsval,
int ns_old, double& m_delt0,
const std::optional<HotRestartState>& initial_state,
std::optional<MultigridInterpolationScheme> interpolation_scheme) {

fc_.fsq = 1.0;

iter2_ = 1;
iter1_ = iter2_;

fc_.ijacob = 0;
fc_.restart_reason = RestartReason::NO_RESTART;
fc_.res0 = -1;
fc_.res1 = -1;
m_delt0 = indata_.delt;

fc_.ns = nsval;

fc_.deltaS = 1.0 / (fc_.ns - 1.0);
fc_.num_surfaces_to_distribute = fc_.ns - 1;
if (fc_.lfreeb) {
fc_.num_surfaces_to_distribute = fc_.ns;
}

int mns = fc_.ns * s_.mnsize;

int irzloff = s_.num_basis * mns;

fc_.neqs = 3 * irzloff;

bool linterp = (ns_old < fc_.ns && ns_old != 0);

if (ns_old != fc_.ns) {

if (linterp && fc_.neqs_old > 0) {
old_xc_scaled_.resize(num_threads_);
old_r_.resize(num_threads_);

for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
old_xc_scaled_[thread_id] = std::make_unique<FourierGeometry>(
&s_, r_[thread_id].get(), fc_.ns_old);
old_r_[thread_id] = std::move(r_[thread_id]);

decomposed_x_[thread_id]->decomposeInto(*old_xc_scaled_[thread_id],
p_[thread_id]->scalxc);
}
}

num_threads_ = vmec_adjust_num_threads(fc_.max_threads(),
fc_.num_surfaces_to_distribute);

if (fc_.lfreeb && fb_vac_.empty()) {
SetupVacuumSolvers();
}

r_.resize(num_threads_);
ls_.resize(num_threads_);
p_.resize(num_threads_);
m_.resize(num_threads_);
decomposed_x_.resize(num_threads_);
physical_x_backup_.resize(num_threads_);
physical_x_.resize(num_threads_);
decomposed_f_.resize(num_threads_);
physical_f_.resize(num_threads_);
decomposed_v_.resize(num_threads_);

for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
r_[thread_id] = std::make_unique<RadialPartitioning>();

const bool printout_radial_partitioning = false;
r_[thread_id]->adjustRadialPartitioning(num_threads_, thread_id, nsval,
fc_.lfreeb,
printout_radial_partitioning);

h_.allocate(*r_[thread_id], fc_.ns);

ls_[thread_id] = std::make_unique<ThreadLocalStorage>(&s_);

p_[thread_id] = std::make_unique<RadialProfiles>(
r_[thread_id].get(), &h_, &indata_, &fc_, indata_.signgs, kPDamp);

p_[thread_id]->setupInputProfiles();

m_[thread_id] = std::make_unique<IdealMhdModel>(
&fc_, &s_, &t_, p_[thread_id].get(), &constants_,
ls_[thread_id].get(), &h_, r_[thread_id].get(), &fb_vac_,
vac_num_threads_, indata_.signgs, indata_.nvacskip,
&vacuum_pressure_state_);
m_[thread_id]->setFromINDATA(indata_.ncurr, indata_.gamma, indata_.tcon0,
indata_.lforbal);
}

absl::Status current_profile_status =
p_[0]->CheckCurrentProfileEnclosesEdgeCurrent();
if (!current_profile_status.ok()) {
return current_profile_status;
}

if (checkpoint == VmecCheckpoint::SPECTRAL_CONSTRAINT &&
iterations_before_checkpointing <= 1) {

return true;
}

for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {

decomposed_x_[thread_id] =
std::make_unique<FourierGeometry>(&s_, r_[thread_id].get(), fc_.ns);

physical_x_[thread_id] =
std::make_unique<FourierGeometry>(&s_, r_[thread_id].get(), fc_.ns);

physical_x_backup_[thread_id] =
std::make_unique<FourierGeometry>(&s_, r_[thread_id].get(), fc_.ns);

physical_f_[thread_id] =
std::make_unique<FourierForces>(&s_, r_[thread_id].get(), fc_.ns);

decomposed_f_[thread_id] =
std::make_unique<FourierForces>(&s_, r_[thread_id].get(), fc_.ns);

decomposed_v_[thread_id] =
std::make_unique<FourierVelocity>(&s_, r_[thread_id].get(), fc_.ns);
}

for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
decomposed_v_[thread_id]->setZero();
decomposed_x_[thread_id]->setZero();
}

for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
p_[thread_id]->evalRadialProfiles(fc_.haveToFlipTheta, constants_);
}

constants_.lamscale = sqrt(constants_.rmsPhiP * fc_.deltaS);

if (checkpoint == VmecCheckpoint::RADIAL_PROFILES_EVAL &&
iterations_before_checkpointing <= 1) {
return true;
}

if (initial_state.has_value() && ns_old == 0) {

for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
if (indata_.lfreeb) {

decomposed_x_[thread_id]->InitFromState(
t_, initial_state->wout.rmnc, initial_state->wout.zmns,
initial_state->wout.lmns_full, initial_state->wout.rmns,
initial_state->wout.zmnc, initial_state->wout.lmnc_full,
*p_[thread_id], constants_, indata_.signgs);
} else {

decomposed_x_[thread_id]->InitFromState(
t_, initial_state->wout.rmnc, initial_state->wout.zmns,
initial_state->wout.lmns_full, initial_state->wout.rmns,
initial_state->wout.zmnc, initial_state->wout.lmnc_full,
*p_[thread_id], constants_, indata_.signgs, &b_);
}
}
} else {
for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
decomposed_x_[thread_id]->interpFromBoundaryAndAxis(t_, b_,
*p_[thread_id]);
}
}
if (checkpoint == VmecCheckpoint::SETUP_INITIAL_STATE &&
iterations_before_checkpointing <= 1) {
return true;
}

if (linterp) {
InterpolateToNextMultigridStep(fc_.ns, fc_.ns_old, p_, r_, old_r_,
decomposed_x_, old_xc_scaled_,
interpolation_scheme);

if (checkpoint == VmecCheckpoint::INTERP) {
return true;
}
}

for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
fc_.restart_reason = RestartReason::NO_RESTART;

RestartIteration(indata_.delt, thread_id);
}

fc_.ns_old = fc_.ns;
fc_.neqs_old = fc_.neqs;
}

return false;
}

absl::StatusOr<bool> Vmec::SolveEquilibrium(
VmecCheckpoint checkpoint, int iterations_before_checkpointing) {

absl::Status status_of_all_threads = absl::OkStatus();
bool any_checkpoint_reached = false;

bool all_errors_are_recoverable = true;

bool liter_flag = true;

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads_)
#endif
{
#ifdef _OPENMP
int thread_id = omp_get_thread_num();
#else
int thread_id = 0;
#endif

bool m_lreset_internal = false;

absl::StatusOr<SolveEqLoopStatus> s = SolveEqLoopStatus::MUST_RETRY;

int n_local_eqsolve_retries = 0;
for (n_local_eqsolve_retries = 0;
n_local_eqsolve_retries < fc_.niterv && s.ok() &&
*s == SolveEqLoopStatus::MUST_RETRY && liter_flag;
n_local_eqsolve_retries++) {

#ifdef _OPENMP
#pragma omp barrier
#endif

s = SolveEquilibriumLoop(
thread_id, iterations_before_checkpointing, checkpoint,
m_lreset_internal,  liter_flag);
}

#ifdef _OPENMP
#pragma omp single nowait
#endif
num_eqsolve_retries_ += n_local_eqsolve_retries;

#ifdef _OPENMP
#pragma omp critical
#endif
{
if (s.ok()) {
any_checkpoint_reached |= (*s == SolveEqLoopStatus::CHECKPOINT_REACHED);
} else {
all_errors_are_recoverable &=
(s.status().code() == absl::StatusCode::kFailedPrecondition);
UpdateStatusForThread(status_of_all_threads, thread_id, s.status());
}
}
}

if (interrupted_) {
return absl::CancelledError("Run interrupted by user");
}

if (!status_of_all_threads.ok()) {
if (indata_.return_outputs_even_if_not_converged &&
all_errors_are_recoverable) {

if (verbose_) {
std::cout << absl::StrFormat(
"WARNING: %s\n"
"return_outputs_even_if_not_converged is set, so returning "
"best-effort (likely unphysical) output for debugging "
"purposes.\n",
status_of_all_threads.message());
}
status_ = VmecStatus::UNRECOVERABLE_ERROR;
return false;
}
return status_of_all_threads;
}

if (!any_checkpoint_reached) {

logger_.EndStage(h_.mhdEnergy * 4.0 * M_PI * M_PI);
}

return any_checkpoint_reached;
}

absl::StatusOr<Vmec::SolveEqLoopStatus> Vmec::SolveEquilibriumLoop(
int thread_id, int iterations_before_checkpointing,
VmecCheckpoint checkpoint, bool& m_lreset_internal, bool& m_liter_flag) {

if (fc_.restart_reason == RestartReason::BAD_JACOBIAN) {
decomposed_x_[thread_id]->setZero();
if (m_lreset_internal) {
decomposed_x_[thread_id]->interpFromBoundaryAndAxis(t_, b_,
*p_[thread_id]);
}

#ifdef _OPENMP
#pragma omp barrier
#endif

#ifdef _OPENMP
#pragma omp single
#endif
{
fc_.restart_reason = RestartReason::NO_RESTART;
}

RestartIteration(fc_.delt0r, thread_id);
}

#ifdef _OPENMP
#pragma omp single
#endif
{

m_liter_flag = true;

status_ = VmecStatus::NORMAL_TERMINATION;
}

for (int force_iteration = iter2_, bad_resets = 0;
(force_iteration <= fc_.niterv) && m_liter_flag; force_iteration++) {
const int iter2 = force_iteration - bad_resets;

absl::StatusOr<bool> reached_checkpoint =
Evolve(checkpoint, iterations_before_checkpointing, fc_.delt0r,
thread_id,  m_liter_flag);
if (!reached_checkpoint.ok()) {
return reached_checkpoint.status();
}
if (*reached_checkpoint) {
return SolveEqLoopStatus::CHECKPOINT_REACHED;
}

if (fc_.ijacob == 0 &&
(status_ == VmecStatus::BAD_JACOBIAN ||
fc_.restart_reason == RestartReason::HUGE_INITIAL_FORCES) &&
fc_.ns >= 3) {

#ifdef _OPENMP
#pragma omp barrier
#endif

#ifdef _OPENMP
#pragma omp single
#endif
{
if (verbose_) {

if (status_ == VmecStatus::BAD_JACOBIAN) {
std::cout << " INITIAL JACOBIAN CHANGED SIGN!\n";
}
std::cout << " TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS\n";
}

b_.RecomputeMagneticAxisToFixJacobianSign(fc_.nsval, indata_.signgs);
fc_.ijacob = 1;

fc_.restart_reason = RestartReason::BAD_JACOBIAN;
}

m_lreset_internal = true;

return SolveEqLoopStatus::MUST_RETRY;
} else if (status_ != VmecStatus::NORMAL_TERMINATION &&
status_ != VmecStatus::SUCCESSFUL_TERMINATION) {

if (!indata_.return_outputs_even_if_not_converged) {
const auto msg = absl::StrFormat(
"FATAL ERROR in thread=%d. The solver failed during the first "
"iterations. This may happen if the initial boundary is poorly "
"shaped or if it isn't spectrally condensed enough.",
thread_id);
return absl::UnknownError(msg);
}

return SolveEqLoopStatus::NORMAL_TERMINATION;
}

if (checkpoint == VmecCheckpoint::EVOLVE &&
iter2 >= iterations_before_checkpointing) {

return SolveEqLoopStatus::CHECKPOINT_REACHED;
}

#ifdef _OPENMP

#pragma omp barrier
#endif

if (fc_.ijacob == 25 || fc_.ijacob == 50) {

#ifdef _OPENMP
#pragma omp single
#endif
{
fc_.restart_reason = RestartReason::BAD_JACOBIAN;
}

RestartIteration(fc_.delt0r, thread_id);

#ifdef _OPENMP
#pragma omp single
#endif
{

const double scale = fc_.ijacob < 50 ? 0.98 : 0.96;

fc_.delt0r = scale * indata_.delt;

if (verbose_) {
std::cout
<< absl::StrFormat(
"HAVING A CONVERGENCE PROBLEM: RESETTING DELT TO %8.3f. "
" If this does NOT resolve the problem,"
" try changing (decrease OR increase) the value of DELT\n",
fc_.delt0r)
<< std::flush;
}

fc_.restart_reason = RestartReason::NO_RESTART;
}

return SolveEqLoopStatus::MUST_RETRY;
} else if (fc_.ijacob >= 75) {

#ifdef _OPENMP
#pragma omp single
#endif
{

status_ = VmecStatus::JACOBIAN_75_TIMES_BAD;
m_liter_flag = false;
}
}

#ifdef _OPENMP
#pragma omp single
#endif
{

if (iter2 == iter1_ || fc_.res0 == -1) {

fc_.res0 = fc_.fsq;
}

fc_.res0 = std::min(fc_.res0, fc_.fsq);

if (indata_.iteration_style == IterationStyle::PARVMEC) {
const double fsq_invariant = fc_.fsqr + fc_.fsqz + fc_.fsql;
if (iter2 == iter1_ || fc_.res1 == -1) {
fc_.res1 = fsq_invariant;
}
fc_.res1 = std::min(fc_.res1, fsq_invariant);
}
}

if (indata_.iteration_style == IterationStyle::PARVMEC) {

const double fsq_invariant = fc_.fsqr + fc_.fsqz + fc_.fsql;
if (fc_.fsq <= fc_.res0 && fsq_invariant <= fc_.res1) {
RestartIteration(fc_.delt0r, thread_id);
} else if ((iter2 - iter1_) > 10 && (fc_.fsq > 1.0e4 * fc_.res0 ||
fsq_invariant > 1.0e4 * fc_.res1)) {
#ifdef _OPENMP
#pragma omp single
#endif
fc_.restart_reason = RestartReason::BAD_PROGRESS;
}
} else if (fc_.fsq <= fc_.res0 && (iter2 - iter1_) > 10) {

RestartIteration(fc_.delt0r, thread_id);
} else if (fc_.fsq > 100.0 * fc_.res0 && iter2 > iter1_) {

#ifdef _OPENMP
#pragma omp single
#endif
fc_.restart_reason = RestartReason::BAD_JACOBIAN;
} else if ((iter2 - iter1_) > fc_.kPreconditionerUpdateInterval / 2 &&
iter2 > 2 * fc_.kPreconditionerUpdateInterval &&
fc_.fsqr + fc_.fsqz > 1.0e-2) {

#ifdef _OPENMP
#pragma omp single
#endif
fc_.restart_reason = RestartReason::BAD_PROGRESS;
}

const RestartReason restart_reason = fc_.restart_reason;

#ifdef _OPENMP
#pragma omp barrier
#endif

if (restart_reason != RestartReason::NO_RESTART) {

RestartIteration(fc_.delt0r, thread_id);

bad_resets++;
#ifdef _OPENMP
#pragma omp single nowait
#endif
iter1_ = iter2;
} else {

if (iter2 % indata_.nstep == 0 || iter2 == 1 || !m_liter_flag) {

physical_x_backup_[thread_id]->ComputeSpectralWidth(t_, *p_[thread_id]);

Printout(fc_.delt0r, thread_id, iter2);

#ifdef _OPENMP
#pragma omp master
#endif
{
if (interrupt_callback_ && interrupt_callback_()) {
m_liter_flag = false;
#pragma omp atomic write
interrupted_ = true;
std::cout << "Received interrupt signal from Python thread.\n";
}
}

if (checkpoint == VmecCheckpoint::PRINTOUT &&
iter2 >= iterations_before_checkpointing) {
return SolveEqLoopStatus::CHECKPOINT_REACHED;
}
}
}

#ifdef _OPENMP
#pragma omp barrier
#endif

#ifdef _OPENMP
#pragma omp single
#endif
{

if (vacuum_pressure_state_ == VacuumPressureState::kInitialized) {
vacuum_pressure_state_ = VacuumPressureState::kActive;

if (verbose_) {
std::cout << absl::StrFormat(
"VACUUM PRESSURE TURNED ON AT %4d ITERATIONS",
iter2_)
<< "\n\n";
}
}
}

#ifdef _OPENMP
#pragma omp atomic write
#endif

iter2_ = (force_iteration - bad_resets) + 1;
}

return SolveEqLoopStatus::NORMAL_TERMINATION;
}

void Vmec::RestartIteration(double& m_delt0r, int thread_id) {
#ifdef _OPENMP
#pragma omp barrier
#endif

if (fc_.restart_reason == RestartReason::BAD_JACOBIAN) {

decomposed_v_[thread_id]->setZero();

*decomposed_x_[thread_id] = *physical_x_backup_[thread_id];

#ifdef _OPENMP
#pragma omp barrier
#endif

#ifdef _OPENMP
#pragma omp single
#endif
{

m_delt0r = m_delt0r * 0.9;

fc_.ijacob = fc_.ijacob + 1;

iter1_ = iter2_;

fc_.restart_reason = RestartReason::NO_RESTART;
}

} else if (fc_.restart_reason == RestartReason::BAD_PROGRESS) {

decomposed_v_[thread_id]->setZero();

*decomposed_x_[thread_id] = *physical_x_backup_[thread_id];

#ifdef _OPENMP
#pragma omp barrier
#endif

#ifdef _OPENMP
#pragma omp single
#endif
{

m_delt0r = m_delt0r / 1.03;

fc_.restart_reason = RestartReason::NO_RESTART;
}
} else {

*physical_x_backup_[thread_id] = *decomposed_x_[thread_id];
}
#ifdef _OPENMP
#pragma omp barrier
#endif
}

absl::StatusOr<bool> Vmec::Evolve(VmecCheckpoint checkpoint,
int iterations_before_checkpointing,
double time_step, int thread_id,
bool& m_liter_flag) {
#ifdef _OPENMP
#pragma omp single
#endif
{
fc_.restart_reason = RestartReason::NO_RESTART;
}

absl::StatusOr<bool> reached_checkpoint = UpdateForwardModel(
checkpoint, iterations_before_checkpointing, thread_id);
if (!reached_checkpoint.ok() || *reached_checkpoint == true) {
return reached_checkpoint;
}

#ifdef _OPENMP
#pragma omp single
#endif
{

bool all_residuals_finite = std::isfinite(fc_.fsqr) &&
std::isfinite(fc_.fsqz) &&
std::isfinite(fc_.fsql);
if ((iter2_ == 1 && fc_.restart_reason == RestartReason::BAD_JACOBIAN) ||
!all_residuals_finite) {

status_ = VmecStatus::BAD_JACOBIAN;
} else if (fc_.fsqr <= fc_.ftolv && fc_.fsqz <= fc_.ftolv &&
fc_.fsql <= fc_.ftolv) {

m_liter_flag = false;
status_ = VmecStatus::SUCCESSFUL_TERMINATION;
}
}

if (status_ != VmecStatus::NORMAL_TERMINATION || !m_liter_flag) {

return false;
}

#ifdef _OPENMP
#pragma omp single
#endif
{

const double fsq1 = fc_.fsqr1 + fc_.fsqz1 + fc_.fsql1;

if (iter2_ == iter1_) {

invTau_.setConstant(0.15 / time_step);
}

{

std::copy(invTau_.data() + 1, invTau_.data() + invTau_.size(),
invTau_.data());
}

if (iter2_ > iter1_) {
double invtau_numerator = 0.;
if (fsq1 != 0.) {

invtau_numerator = std::min(std::abs(std::log(fsq1 / fc_.fsq)), 0.15);
}

invTau_[invTau_.size() - 1] = invtau_numerator / time_step;
}

fc_.fsq = fsq1;
}

if (r_[thread_id]->nsMaxF1 == fc_.ns) {
fc_.force_residual_r.push_back(fc_.fsqr);
fc_.force_residual_z.push_back(fc_.fsqz);
fc_.force_residual_lambda.push_back(fc_.fsql);
if (fc_.lfreeb) {

fc_.delbsq.push_back(m_[thread_id]->get_delbsq());
} else {
fc_.delbsq.push_back(0.0);
}
fc_.restart_reasons.push_back(fc_.restart_reason);
fc_.mhd_energy.push_back(h_.mhdEnergy);
}

const double otav = invTau_.sum() / kNDamp;

const double dtau = time_step * otav / 2.0;
const double b1 = 1.0 - dtau;
const double fac = 1.0 / (1.0 + dtau);

PerformTimeStep(fac, b1, time_step, thread_id);

return false;
}

void Vmec::Printout(double delt0r, int thread_id, int iter2) {
#ifdef _OPENMP
#pragma omp single
#endif
{
h_.ResetSpectralWidthAccumulators();
}
p_[thread_id]->AccumulateVolumeAveragedSpectralWidth();
#ifdef _OPENMP
#pragma omp barrier
#endif

if (r_[thread_id]->nsMaxF1 == fc_.ns) {

const GeometricOffset& geometric_offset = h_.GetGeometricOffset();
double r00 = geometric_offset.r_00;

double energy = h_.mhdEnergy * 4.0 * M_PI * M_PI;

double betaVolAvg = h_.thermalEnergy / h_.magneticEnergy;

double volAvgM = h_.VolumeAveragedSpectralWidth();

double delbsq = m_[thread_id]->get_delbsq();

logger_.LogIteration(iter2, fc_.fsqr, fc_.fsqz, fc_.fsql, fc_.fsqr1,
fc_.fsqz1, fc_.fsql1, delt0r, r00, energy, betaVolAvg,
volAvgM, delbsq);
}
}

absl::StatusOr<bool> Vmec::UpdateForwardModel(
VmecCheckpoint checkpoint, int iterations_before_checkpointing,
int thread_id) {
bool need_restart = false;

absl::StatusOr<bool> reached_checkpoint = m_[thread_id]->update(
*decomposed_x_[thread_id], *physical_x_[thread_id],
*decomposed_f_[thread_id], *physical_f_[thread_id], need_restart,
last_preconditioner_update_, last_full_update_nestor_, fc_, iter1_,
iter2_, checkpoint, iterations_before_checkpointing, verbose_);
if (!reached_checkpoint.ok()) {
return reached_checkpoint;
}

if (need_restart) {
double delt0 = indata_.delt;
RestartIteration(delt0, thread_id);

#ifdef _OPENMP
#pragma omp single nowait
#endif

fc_.restart_reason = RestartReason::NO_RESTART;
}

#ifdef _OPENMP
#pragma omp barrier
#endif

return reached_checkpoint;
}

void Vmec::PerformTimeStep(double fac, double b1, double time_step,
int thread_id) {
#ifdef _OPENMP
#pragma omp barrier
#endif

performTimeStep(s_, fc_, *r_[thread_id], fac, b1, time_step,
*decomposed_x_[thread_id],
*decomposed_v_[thread_id],
*decomposed_f_[thread_id],
h_);

#ifdef _OPENMP
#pragma omp barrier
#endif
}

void Vmec::performTimeStep(const Sizes& s, const FlowControl& fc,
const RadialPartitioning& r, double velocity_scale,
double conjugation_parameter, double time_step,
FourierGeometry& m_decomposed_x,
FourierVelocity& m_decomposed_v,
const FourierForces& decomposed_f,
HandoverStorage& m_h_) const {

for (int jF = r.nsMinF; jF < r.nsMaxFIncludingLcfs; ++jF) {
for (int m = 0; m < s.mpol; ++m) {
for (int n = 0; n < s.ntor + 1; ++n) {
const int idx_mn = ((jF - r.nsMinF) * s.mpol + m) * (s.ntor + 1) + n;
const int idx_mn1 = ((jF - r.nsMinF1) * s.mpol + m) * (s.ntor + 1) + n;

m_decomposed_v.vrcc[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vrcc[idx_mn] +
time_step * decomposed_f.frcc[idx_mn]);
m_decomposed_v.vzsc[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vzsc[idx_mn] +
time_step * decomposed_f.fzsc[idx_mn]);
m_decomposed_v.vlsc[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vlsc[idx_mn] +
time_step * decomposed_f.flsc[idx_mn]);
if (s.lthreed) {
m_decomposed_v.vrss[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vrss[idx_mn] +
time_step * decomposed_f.frss[idx_mn]);
m_decomposed_v.vzcs[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vzcs[idx_mn] +
time_step * decomposed_f.fzcs[idx_mn]);
m_decomposed_v.vlcs[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vlcs[idx_mn] +
time_step * decomposed_f.flcs[idx_mn]);
}
if (s.lasym) {
m_decomposed_v.vrsc[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vrsc[idx_mn] +
time_step * decomposed_f.frsc[idx_mn]);
m_decomposed_v.vzcc[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vzcc[idx_mn] +
time_step * decomposed_f.fzcc[idx_mn]);
m_decomposed_v.vlcc[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vlcc[idx_mn] +
time_step * decomposed_f.flcc[idx_mn]);
if (s.lthreed) {
m_decomposed_v.vrcs[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vrcs[idx_mn] +
time_step * decomposed_f.frcs[idx_mn]);
m_decomposed_v.vzss[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vzss[idx_mn] +
time_step * decomposed_f.fzss[idx_mn]);
m_decomposed_v.vlss[idx_mn] =
velocity_scale *
(conjugation_parameter * m_decomposed_v.vlss[idx_mn] +
time_step * decomposed_f.flss[idx_mn]);
}
}

m_decomposed_x.rmncc[idx_mn1] +=
time_step * m_decomposed_v.vrcc[idx_mn];
m_decomposed_x.zmnsc[idx_mn1] +=
time_step * m_decomposed_v.vzsc[idx_mn];
m_decomposed_x.lmnsc[idx_mn1] +=
time_step * m_decomposed_v.vlsc[idx_mn];
if (s.lthreed) {
m_decomposed_x.rmnss[idx_mn1] +=
time_step * m_decomposed_v.vrss[idx_mn];
m_decomposed_x.zmncs[idx_mn1] +=
time_step * m_decomposed_v.vzcs[idx_mn];
m_decomposed_x.lmncs[idx_mn1] +=
time_step * m_decomposed_v.vlcs[idx_mn];
}
if (s.lasym) {
m_decomposed_x.rmnsc[idx_mn1] +=
time_step * m_decomposed_v.vrsc[idx_mn];
m_decomposed_x.zmncc[idx_mn1] +=
time_step * m_decomposed_v.vzcc[idx_mn];
m_decomposed_x.lmncc[idx_mn1] +=
time_step * m_decomposed_v.vlcc[idx_mn];
if (s.lthreed) {
m_decomposed_x.rmncs[idx_mn1] +=
time_step * m_decomposed_v.vrcs[idx_mn];
m_decomposed_x.zmnss[idx_mn1] +=
time_step * m_decomposed_v.vzss[idx_mn];
m_decomposed_x.lmnss[idx_mn1] +=
time_step * m_decomposed_v.vlss[idx_mn];
}
}
}
}
}

bool hasInside = (r.nsMinF1 > 0);
bool hasOutside = (r.nsMaxF1 < fc.ns);

if (hasInside) {

const int prev_thread = r.get_thread_id() - 1;
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMinF - r.nsMinF1) * s_.mnsize + mn;
m_h_.rmncc_o(prev_thread, mn) = m_decomposed_x.rmncc[idx_mn];
m_h_.zmnsc_o(prev_thread, mn) = m_decomposed_x.zmnsc[idx_mn];
m_h_.lmnsc_o(prev_thread, mn) = m_decomposed_x.lmnsc[idx_mn];
}

if (s_.lthreed) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMinF - r.nsMinF1) * s_.mnsize + mn;
m_h_.rmnss_o(prev_thread, mn) = m_decomposed_x.rmnss[idx_mn];
m_h_.zmncs_o(prev_thread, mn) = m_decomposed_x.zmncs[idx_mn];
m_h_.lmncs_o(prev_thread, mn) = m_decomposed_x.lmncs[idx_mn];
}
}

if (s_.lasym) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMinF - r.nsMinF1) * s_.mnsize + mn;
m_h_.rmnsc_o(prev_thread, mn) = m_decomposed_x.rmnsc[idx_mn];
m_h_.zmncc_o(prev_thread, mn) = m_decomposed_x.zmncc[idx_mn];
m_h_.lmncc_o(prev_thread, mn) = m_decomposed_x.lmncc[idx_mn];
}

if (s_.lthreed) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMinF - r.nsMinF1) * s_.mnsize + mn;
m_h_.rmncs_o(prev_thread, mn) = m_decomposed_x.rmncs[idx_mn];
m_h_.zmnss_o(prev_thread, mn) = m_decomposed_x.zmnss[idx_mn];
m_h_.lmnss_o(prev_thread, mn) = m_decomposed_x.lmnss[idx_mn];
}
}
}
}

if (hasOutside) {

const int next_thread = r.get_thread_id() + 1;
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMaxF - 1 - r.nsMinF1) * s_.mnsize + mn;
m_h_.rmncc_i(next_thread, mn) = m_decomposed_x.rmncc[idx_mn];
m_h_.zmnsc_i(next_thread, mn) = m_decomposed_x.zmnsc[idx_mn];
m_h_.lmnsc_i(next_thread, mn) = m_decomposed_x.lmnsc[idx_mn];
}

if (s_.lthreed) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMaxF - 1 - r.nsMinF1) * s_.mnsize + mn;
m_h_.rmnss_i(next_thread, mn) = m_decomposed_x.rmnss[idx_mn];
m_h_.zmncs_i(next_thread, mn) = m_decomposed_x.zmncs[idx_mn];
m_h_.lmncs_i(next_thread, mn) = m_decomposed_x.lmncs[idx_mn];
}
}

if (s_.lasym) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMaxF - 1 - r.nsMinF1) * s_.mnsize + mn;
m_h_.rmnsc_i(next_thread, mn) = m_decomposed_x.rmnsc[idx_mn];
m_h_.zmncc_i(next_thread, mn) = m_decomposed_x.zmncc[idx_mn];
m_h_.lmncc_i(next_thread, mn) = m_decomposed_x.lmncc[idx_mn];
}

if (s_.lthreed) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMaxF - 1 - r.nsMinF1) * s_.mnsize + mn;
m_h_.rmncs_i(next_thread, mn) = m_decomposed_x.rmncs[idx_mn];
m_h_.zmnss_i(next_thread, mn) = m_decomposed_x.zmnss[idx_mn];
m_h_.lmnss_i(next_thread, mn) = m_decomposed_x.lmnss[idx_mn];
}
}
}
}

#ifdef _OPENMP
#pragma omp barrier
#endif

if (hasOutside) {

const int this_thread = r.get_thread_id();
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMaxF1 - 1 - r.nsMinF1) * s_.mnsize + mn;
m_decomposed_x.rmncc[idx_mn] = m_h_.rmncc_o(this_thread, mn);
m_decomposed_x.zmnsc[idx_mn] = m_h_.zmnsc_o(this_thread, mn);
m_decomposed_x.lmnsc[idx_mn] = m_h_.lmnsc_o(this_thread, mn);
}

if (s_.lthreed) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMaxF1 - 1 - r.nsMinF1) * s_.mnsize + mn;
m_decomposed_x.rmnss[idx_mn] = m_h_.rmnss_o(this_thread, mn);
m_decomposed_x.zmncs[idx_mn] = m_h_.zmncs_o(this_thread, mn);
m_decomposed_x.lmncs[idx_mn] = m_h_.lmncs_o(this_thread, mn);
}
}

if (s_.lasym) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMaxF1 - 1 - r.nsMinF1) * s_.mnsize + mn;
m_decomposed_x.rmnsc[idx_mn] = m_h_.rmnsc_o(this_thread, mn);
m_decomposed_x.zmncc[idx_mn] = m_h_.zmncc_o(this_thread, mn);
m_decomposed_x.lmncc[idx_mn] = m_h_.lmncc_o(this_thread, mn);
}

if (s_.lthreed) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMaxF1 - 1 - r.nsMinF1) * s_.mnsize + mn;
m_decomposed_x.rmncs[idx_mn] = m_h_.rmncs_o(this_thread, mn);
m_decomposed_x.zmnss[idx_mn] = m_h_.zmnss_o(this_thread, mn);
m_decomposed_x.lmnss[idx_mn] = m_h_.lmnss_o(this_thread, mn);
}
}
}
}

if (hasInside) {

const int this_thread = r.get_thread_id();
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMinF1 - r.nsMinF1) * s_.mnsize + mn;
m_decomposed_x.rmncc[idx_mn] = m_h_.rmncc_i(this_thread, mn);
m_decomposed_x.zmnsc[idx_mn] = m_h_.zmnsc_i(this_thread, mn);
m_decomposed_x.lmnsc[idx_mn] = m_h_.lmnsc_i(this_thread, mn);
}

if (s_.lthreed) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMinF1 - r.nsMinF1) * s_.mnsize + mn;
m_decomposed_x.rmnss[idx_mn] = m_h_.rmnss_i(this_thread, mn);
m_decomposed_x.zmncs[idx_mn] = m_h_.zmncs_i(this_thread, mn);
m_decomposed_x.lmncs[idx_mn] = m_h_.lmncs_i(this_thread, mn);
}
}

if (s_.lasym) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMinF1 - r.nsMinF1) * s_.mnsize + mn;
m_decomposed_x.rmnsc[idx_mn] = m_h_.rmnsc_i(this_thread, mn);
m_decomposed_x.zmncc[idx_mn] = m_h_.zmncc_i(this_thread, mn);
m_decomposed_x.lmncc[idx_mn] = m_h_.lmncc_i(this_thread, mn);
}

if (s_.lthreed) {
for (int mn = 0; mn < s_.mnsize; ++mn) {
int idx_mn = (r.nsMinF1 - r.nsMinF1) * s_.mnsize + mn;
m_decomposed_x.rmncs[idx_mn] = m_h_.rmncs_i(this_thread, mn);
m_decomposed_x.zmnss[idx_mn] = m_h_.zmnss_i(this_thread, mn);
m_decomposed_x.lmnss[idx_mn] = m_h_.lmnss_i(this_thread, mn);
}
}
}
}

#ifdef _OPENMP
#pragma omp barrier
#endif
}

void Vmec::InterpolateToNextMultigridStep(
int ns_new, int ns_old,
const std::vector<std::unique_ptr<RadialProfiles>>& p,
const std::vector<std::unique_ptr<RadialPartitioning>>& r_new,
const std::vector<std::unique_ptr<RadialPartitioning>>& r_old,
std::vector<std::unique_ptr<FourierGeometry>>& m_x_new,
std::vector<std::unique_ptr<FourierGeometry>>& m_x_old,
std::optional<MultigridInterpolationScheme> interpolation_scheme) {

const double hs_old = 1.0 / (ns_old - 1.0);

const int num_threads_new = static_cast<int>(r_new.size());
const int num_threads_old = static_cast<int>(r_old.size());

int thread_with_ns_0 = 0;
int thread_with_ns_1 = 0;
int thread_with_ns_2 = 0;
for (int thread_id = 0; thread_id < num_threads_old; ++thread_id) {
const int nsMinF = r_old[thread_id]->nsMinF;
const int nsMaxFIncludingLcfs = r_old[thread_id]->nsMaxFIncludingLcfs;
if (nsMinF <= 0 && 0 < nsMaxFIncludingLcfs) {
thread_with_ns_0 = thread_id;
}
if (nsMinF <= 1 && 1 < nsMaxFIncludingLcfs) {
thread_with_ns_1 = thread_id;
}
if (nsMinF <= 2 && 2 < nsMaxFIncludingLcfs) {
thread_with_ns_2 = thread_id;
}
}

for (int m = 1; m < s_.mpol; m += 2) {
for (int n = 0; n < s_.ntor + 1; ++n) {
const int idx_fc_0 =
((0 - m_x_old[thread_with_ns_0]->nsMin()) * s_.mpol + m) *
(s_.ntor + 1) +
n;
const int idx_fc_1 =
((1 - m_x_old[thread_with_ns_1]->nsMin()) * s_.mpol + m) *
(s_.ntor + 1) +
n;
const int idx_fc_2 =
((2 - m_x_old[thread_with_ns_2]->nsMin()) * s_.mpol + m) *
(s_.ntor + 1) +
n;

m_x_old[thread_with_ns_0]->rmncc[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->rmncc[idx_fc_1] -
m_x_old[thread_with_ns_2]->rmncc[idx_fc_2];
m_x_old[thread_with_ns_0]->zmnsc[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->zmnsc[idx_fc_1] -
m_x_old[thread_with_ns_2]->zmnsc[idx_fc_2];
m_x_old[thread_with_ns_0]->lmnsc[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->lmnsc[idx_fc_1] -
m_x_old[thread_with_ns_2]->lmnsc[idx_fc_2];
if (s_.lthreed) {
m_x_old[thread_with_ns_0]->rmnss[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->rmnss[idx_fc_1] -
m_x_old[thread_with_ns_2]->rmnss[idx_fc_2];
m_x_old[thread_with_ns_0]->zmncs[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->zmncs[idx_fc_1] -
m_x_old[thread_with_ns_2]->zmncs[idx_fc_2];
m_x_old[thread_with_ns_0]->lmncs[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->lmncs[idx_fc_1] -
m_x_old[thread_with_ns_2]->lmncs[idx_fc_2];
}
if (s_.lasym) {
m_x_old[thread_with_ns_0]->rmnsc[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->rmnsc[idx_fc_1] -
m_x_old[thread_with_ns_2]->rmnsc[idx_fc_2];
m_x_old[thread_with_ns_0]->zmncc[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->zmncc[idx_fc_1] -
m_x_old[thread_with_ns_2]->zmncc[idx_fc_2];
m_x_old[thread_with_ns_0]->lmncc[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->lmncc[idx_fc_1] -
m_x_old[thread_with_ns_2]->lmncc[idx_fc_2];
if (s_.lthreed) {
m_x_old[thread_with_ns_0]->rmncs[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->rmncs[idx_fc_1] -
m_x_old[thread_with_ns_2]->rmncs[idx_fc_2];
m_x_old[thread_with_ns_0]->zmnss[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->zmnss[idx_fc_1] -
m_x_old[thread_with_ns_2]->zmnss[idx_fc_2];
m_x_old[thread_with_ns_0]->lmnss[idx_fc_0] =
2.0 * m_x_old[thread_with_ns_1]->lmnss[idx_fc_1] -
m_x_old[thread_with_ns_2]->lmnss[idx_fc_2];
}
}
}
}

std::vector<std::span<double> FourierGeometry::*> coefficient_members = {
&FourierGeometry::rmncc, &FourierGeometry::zmnsc,
&FourierGeometry::lmnsc};
if (s_.lthreed) {
coefficient_members.push_back(&FourierGeometry::rmnss);
coefficient_members.push_back(&FourierGeometry::zmncs);
coefficient_members.push_back(&FourierGeometry::lmncs);
}
if (s_.lasym) {
coefficient_members.push_back(&FourierGeometry::rmnsc);
coefficient_members.push_back(&FourierGeometry::zmncc);
coefficient_members.push_back(&FourierGeometry::lmncc);
if (s_.lthreed) {
coefficient_members.push_back(&FourierGeometry::rmncs);
coefficient_members.push_back(&FourierGeometry::zmnss);
coefficient_members.push_back(&FourierGeometry::lmnss);
}
}

const MultigridInterpolationScheme scheme =
interpolation_scheme.value_or(MultigridInterpolationScheme::kLinear);

for (int thread_id = 0; thread_id < num_threads_new; ++thread_id) {
for (int jNew = r_new[thread_id]->nsMinF1; jNew < r_new[thread_id]->nsMaxF1;
++jNew) {
const double sj = jNew / (ns_new - 1.0);

const int js1 = (jNew * (ns_old - 1)) / (ns_new - 1);
const int js2 = std::min(js1 + 1, ns_old - 1);

std::array<int, 4> stencil{};
std::array<double, 4> weight{};
int stencil_size = 0;

if (scheme == MultigridInterpolationScheme::kLinear || ns_old < 4) {
double xint = (sj - js1 * hs_old) / hs_old;
xint = std::min(1.0, std::max(0.0, xint));
stencil = {js1, js2, 0, 0};
weight = {1.0 - xint, xint, 0.0, 0.0};
stencil_size = 2;
} else {

const int base = std::min(std::max(js1 - 1, 0), ns_old - 4);
std::array<double, 4> node{};
double target = sj;
for (int k = 0; k < 4; ++k) {
stencil[k] = base + k;
node[k] = stencil[k] * hs_old;
}
if (scheme == MultigridInterpolationScheme::kCubicRho) {
target = std::sqrt(sj);
for (int k = 0; k < 4; ++k) {
node[k] = std::sqrt(node[k]);
}
}
for (int k = 0; k < 4; ++k) {
double w = 1.0;
for (int l = 0; l < 4; ++l) {
if (l != k) {
w *= (target - node[l]) / (node[k] - node[l]);
}
}
weight[k] = w;
}
stencil_size = 4;
}

std::array<int, 4> source_thread{};
for (int k = 0; k < stencil_size; ++k) {
for (int old_thread_id = 0; old_thread_id < num_threads_old;
++old_thread_id) {
if (r_old[old_thread_id]->nsMinF1 <= stencil[k] &&
stencil[k] < r_old[old_thread_id]->nsMaxF1) {
source_thread[k] = old_thread_id;
}
}
}

for (int m = 0; m < s_.mpol; ++m) {
const int m_parity = m % 2;
const double scalxc =
p[thread_id]
->scalxc[(jNew - r_new[thread_id]->nsMinF1) * 2 + m_parity];

for (int n = 0; n < s_.ntor + 1; ++n) {
const int idx_fc_jNew =
((jNew - m_x_new[thread_id]->nsMin()) * s_.mpol + m) *
(s_.ntor + 1) +
n;

for (const auto member : coefficient_members) {
double value = 0.0;
for (int k = 0; k < stencil_size; ++k) {
const FourierGeometry& source = *m_x_old[source_thread[k]];
const int idx_fc_k =
((stencil[k] - source.nsMin()) * s_.mpol + m) *
(s_.ntor + 1) +
n;
value += weight[k] * (source.*member)[idx_fc_k];
}
((*m_x_new[thread_id]).*member)[idx_fc_jNew] = value / scalxc;
}
}
}
}
}

for (int m = 1; m < s_.mpol; m += 2) {
for (int n = 0; n < s_.ntor + 1; ++n) {
const int idx_fc_0 =
((0 - m_x_old[thread_with_ns_0]->nsMin()) * s_.mpol + m) *
(s_.ntor + 1) +
n;

m_x_new[thread_with_ns_0]->rmncc[idx_fc_0] = 0.0;
m_x_new[thread_with_ns_0]->zmnsc[idx_fc_0] = 0.0;
m_x_new[thread_with_ns_0]->lmnsc[idx_fc_0] = 0.0;
if (s_.lthreed) {
m_x_new[thread_with_ns_0]->rmnss[idx_fc_0] = 0.0;
m_x_new[thread_with_ns_0]->zmncs[idx_fc_0] = 0.0;
m_x_new[thread_with_ns_0]->lmncs[idx_fc_0] = 0.0;
}
if (s_.lasym) {
m_x_new[thread_with_ns_0]->rmnsc[idx_fc_0] = 0.0;
m_x_new[thread_with_ns_0]->zmncc[idx_fc_0] = 0.0;
m_x_new[thread_with_ns_0]->lmncc[idx_fc_0] = 0.0;
if (s_.lthreed) {
m_x_new[thread_with_ns_0]->rmncs[idx_fc_0] = 0.0;
m_x_new[thread_with_ns_0]->zmnss[idx_fc_0] = 0.0;
m_x_new[thread_with_ns_0]->lmnss[idx_fc_0] = 0.0;
}
}
}
}
}

}

// source: vmecpp/vmec/vmec_constants/vmec_constants.cc

vmecpp::VmecConstants::VmecConstants() { reset(); }

void vmecpp::VmecConstants::reset() {
rmsPhiP = 0.;
lamscale = 0.;
}

// source: vmecpp/vmec/vmec_standalone/vmec_standalone.cc

#include <iostream>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"

using vmecpp::OutputQuantities;
using vmecpp::VmecINDATA;

using file_io::ReadFile;

int main(int argc, char **argv) {
if (argc < 2 || argc > 3) {
std::cerr << "usage: " << argv[0] << " input_file.json [n_max_threads]\n";
return 1;
}

absl::StatusOr<std::string> indata_json = ReadFile(argv[1]);
if (!indata_json.ok()) {
std::cerr << "Could not read input file '" << argv[1]
<< "': " << indata_json.status() << "\n";
return 1;
}

absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
if (!vmec_indata.ok()) {
std::cerr << "Could not parse input file '" << argv[1]
<< "' into VmecINDATA: " << vmec_indata.status() << "\n";
return 1;
}

std::optional<int> max_threads = std::nullopt;
if (argc == 3) {
max_threads = std::atoi(argv[2]);
}

const absl::StatusOr<OutputQuantities> out =
vmecpp::run(*vmec_indata,  std::nullopt,
max_threads);
if (!out.ok()) {
std::cerr << "Error encountered during the VMEC++ run: " << out.status()
<< "\n";
return 1;
}

const std::string out_path =
absl::StrCat(absl::StripSuffix(argv[1], ".json"), ".out.h5");
const absl::Status status = out->Save(out_path);
if (!status.ok()) {
std::cerr << "Error encountered writing the output file '" << out_path
<< "': " << status << "\n";
return 1;
}

return 0;
}
