#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <omp.h>

extern int size, center, num_pix ;
extern uint8_t *mask ;

void make_rot_quat(double *quaternion, double rot[3][3]) {
	double q0, q1, q2, q3, q01, q02, q03, q11, q12, q13, q22, q23, q33 ;
	
	q0 = quaternion[0] ;
	q1 = quaternion[1] ;
	q2 = quaternion[2] ;
	q3 = quaternion[3] ;
	
	q01 = q0*q1 ;
	q02 = q0*q2 ;
	q03 = q0*q3 ;
	q11 = q1*q1 ;
	q12 = q1*q2 ;
	q13 = q1*q3 ;
	q22 = q2*q2 ;
	q23 = q2*q3 ;
	q33 = q3*q3 ;
	
	rot[0][0] = (1. - 2.*(q22 + q33)) ;
	rot[0][1] = 2.*(q12 + q03) ;
	rot[0][2] = 2.*(q13 - q02) ;
	rot[1][0] = 2.*(q12 - q03) ;
	rot[1][1] = (1. - 2.*(q11 + q33)) ;
	rot[1][2] = 2.*(q01 + q23) ;
	rot[2][0] = 2.*(q02 + q13) ;
	rot[2][1] = 2.*(q23 - q01) ;
	rot[2][2] = (1. - 2.*(q11 + q22)) ;
}

/*
Generates slice[t] from model3d[x] by interpolation at angle theta
The locations of the pixels in slice[t] are given by detector[t]
Global variables required:
	size, center, num_pix ;
*/
void slice_gen(double *quaternion, double rescale, double slice[], double model3d[], double detector[]) {
	int t, i, j, x, y, z ;
	double tx, ty, tz, fx, fy, fz, cx, cy, cz ;
	double rot_pix[3], rot[3][3] = {{0}} ;
	
	make_rot_quat(quaternion, rot) ;
	
	for (t = 0 ; t < num_pix ; ++t) {
		for (i = 0 ; i < 3 ; ++i) {
			rot_pix[i] = 0. ;
			for (j = 0 ; j < 3 ; ++j) 
				rot_pix[i] += rot[i][j] * detector[t*4 + j] ;
			rot_pix[i] += center ;
		}
		
		tx = rot_pix[0] ;
		ty = rot_pix[1] ;
		tz = rot_pix[2] ;
		
		x = tx ;
		y = ty ;
		z = tz ;
		
		if (x < 0 || x > size-2 || y < 0 || y > size-2 || z < 0 || z > size-2) {
			slice[t] = DBL_MIN ;
			continue ;
		}
		
		fx = tx - x ;
		fy = ty - y ;
		fz = tz - z ;
		cx = 1. - fx ;
		cy = 1. - fy ;
		cz = 1. - fz ;
		
		slice[t] = cx*cy*cz*model3d[x*size*size + y*size + z] +
		           cx*cy*fz*model3d[x*size*size + y*size + ((z+1)%size)] +
		           cx*fy*cz*model3d[x*size*size + ((y+1)%size)*size + z] +
		           cx*fy*fz*model3d[x*size*size + ((y+1)%size)*size + ((z+1)%size)] +
		           fx*cy*cz*model3d[((x+1)%size)*size*size + y*size + z] +
		           fx*cy*fz*model3d[((x+1)%size)*size*size + y*size + ((z+1)%size)] + 
		           fx*fy*cz*model3d[((x+1)%size)*size*size + ((y+1)%size)*size + z] + 
		           fx*fy*fz*model3d[((x+1)%size)*size*size + ((y+1)%size)*size + ((z+1)%size)] ;
		
		// Correct for solid angle and polarization
		slice[t] *= detector[t*4 + 3] ;
		
		if (slice[t] <= 0.)
			slice[t] = DBL_MIN ;
		
		// Use rescale as flag on whether to take log or not
		else if (rescale != 0.) 
			slice[t] = log(slice[t] * rescale) ;
	}
}

/* 
Merges slice[t] into model3d[x] at the given angle
Also adds to weight[x] containing the interpolation weights
The locations of the pixels in slice[t] are given by detector[t]
Global variables required:
	size, center, num_pix ;
*/
void slice_merge(double *quaternion, double slice[], double model3d[], double weight[], double detector[]) {
	int t, i, j, x, y, z ;
	double tx, ty, tz, fx, fy, fz, cx, cy, cz, w, f ;
	double rot_pix[3], rot[3][3] = {{0}} ;
	
	make_rot_quat(quaternion, rot) ;
	
	for (t = 0 ; t < num_pix ; ++t) {
		if (mask[t] > 1)
			continue ;
		
		for (i = 0 ; i < 3 ; ++i) {
			rot_pix[i] = 0. ;
			for (j = 0 ; j < 3 ; ++j)
				rot_pix[i] += rot[i][j] * detector[t*4 + j] ;
			rot_pix[i] += center ;
		}
		
		tx = rot_pix[0] ;
		ty = rot_pix[1] ;
		tz = rot_pix[2] ;
		
		x = tx ;
		y = ty ;
		z = tz ;
		
		if (x < 0 || x > size-2 || y < 0 || y > size-2 || z < 0 || z > size-2)
			continue ;
		
		fx = tx - x ;
		fy = ty - y ;
		fz = tz - z ;
		cx = 1. - fx ;
		cy = 1. - fy ;
		cz = 1. - fz ;
		
		// Correct for solid angle and polarization
		slice[t] /= detector[t*4 + 3] ;
		w = slice[t] ;
		
		f = cx*cy*cz ;
		weight[x*size*size + y*size + z] += f ;
		model3d[x*size*size + y*size + z] += f * w ;
		
		f = cx*cy*fz ;
		weight[x*size*size + y*size + ((z+1)%size)] += f ;
		model3d[x*size*size + y*size + ((z+1)%size)] += f * w ;
		
		f = cx*fy*cz ;
		weight[x*size*size + ((y+1)%size)*size + z] += f ;
		model3d[x*size*size + ((y+1)%size)*size + z] += f * w ;
		
		f = cx*fy*fz ;
		weight[x*size*size + ((y+1)%size)*size + ((z+1)%size)] += f ;
		model3d[x*size*size + ((y+1)%size)*size + ((z+1)%size)] += f * w ;
		
		f = fx*cy*cz ;
		weight[((x+1)%size)*size*size + y*size + z] += f ;
		model3d[((x+1)%size)*size*size + y*size + z] += f * w ;
		
		f = fx*cy*fz ;
		weight[((x+1)%size)*size*size + y*size + ((z+1)%size)] += f ;
		model3d[((x+1)%size)*size*size + y*size + ((z+1)%size)] += f * w ;
		
		f = fx*fy*cz ;
		weight[((x+1)%size)*size*size + ((y+1)%size)*size + z] += f ;
		model3d[((x+1)%size)*size*size + ((y+1)%size)*size + z] += f * w ;
		
		f = fx*fy*fz ;
		weight[((x+1)%size)*size*size + ((y+1)%size)*size + ((z+1)%size)] += f ;
		model3d[((x+1)%size)*size*size + ((y+1)%size)*size + ((z+1)%size)] += f * w ;
	}
}

/* Rotates cubic model according to given rotation matrix
 * 	Adds to rotated model. Does not zero output model.
 * 	Arguments:
 * 		rot[3][3] - Rotation matrix
 * 		m - Pointer to model to rotate
 * 		s - Size of model. Center assumed to be at (s/2, s/2, s/2)
 * 		rotmodel - Pointer to rotated model
 */
void rotate_model(double rot[3][3], double *m, int s, double *rotmodel) {
	#pragma omp parallel default(shared)
	{
		int x, y, z, i, c = s/2, vx, vy, vz ;
		double fx, fy, fz, cx, cy, cz ;
		double rot_vox[3] ;
		
		#pragma omp for schedule(static,1)
		for (vx = -c ; vx < s-c-1 ; ++vx)
		for (vy = -c ; vy < s-c-1 ; ++vy)
		for (vz = -c ; vz < s-c-1 ; ++vz) {
			for (i = 0 ; i < 3 ; ++i) {
				rot_vox[i] = 0. ;
				rot_vox[i] += rot[i][0]*vx + rot[i][1]*vy + rot[i][2]*vz ;
				rot_vox[i] += c ;
			}
			
			if (rot_vox[0] < 0 || rot_vox[0] >= s - 1) continue ;
			if (rot_vox[1] < 0 || rot_vox[1] >= s - 1) continue ;
			if (rot_vox[2] < 0 || rot_vox[2] >= s - 1) continue ;
			
			x = (int) rot_vox[0] ;
			y = (int) rot_vox[1] ;
			z = (int) rot_vox[2] ;
			fx = rot_vox[0] - x ;
			fy = rot_vox[1] - y ;
			fz = rot_vox[2] - z ;
			cx = 1. - fx ;
			cy = 1. - fy ;
			cz = 1. - fz ;
			
			rotmodel[(vx+c)*s*s + (vy+c)*s + (vz+c)] +=
				cx*cy*cz*m[x*s*s + y*s + z] + 
				cx*cy*fz*m[x*s*s + y*s + ((z+1)%s)] + 
				cx*fy*cz*m[x*s*s + ((y+1)%s)*s + z] + 
				cx*fy*fz*m[x*s*s + ((y+1)%s)*s + ((z+1)%s)] + 
				fx*cy*cz*m[((x+1)%s)*s*s + y*s + z] +
				fx*cy*fz*m[((x+1)%s)*s*s + y*s + ((z+1)%s)] +
				fx*fy*cz*m[((x+1)%s)*s*s + ((y+1)%s)*s + z] +
				fx*fy*fz*m[((x+1)%s)*s*s + ((y+1)%s)*s + ((z+1)%s)] ;
		}
	}
}

static double icos_list[60][3][3] = {
	{{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}},
	{{-1., 0., 0.}, {0., -1., 0.}, {0., 0., 1.}},
	{{0., 0., 1.}, {1., 0., 0.}, {0., 1., 0.}},
	{{0., 1., 0.}, {0., 0., 1.}, {1., 0., 0.}},
	{{0., -1., 0.}, {0., 0., -1.}, {1., 0., 0.}},
	{{0., 0., 1.}, {-1., 0., 0.}, {0., -1., 0.}},
	{{0., 1., 0.}, {0., 0., -1.}, {-1., 0., 0.}},
	{{0., 0., -1.}, {-1., 0., 0.}, {0., 1., 0.}},
	{{1., 0., 0.}, {0., -1., 0.}, {0., 0., -1.}},
	{{0., 0., -1.}, {1., 0., 0.}, {0., -1., 0.}},
	{{0., -1., 0.}, {0., 0., 1.}, {-1., 0., 0.}},
	{{-1., 0., 0.}, {0., 1., 0.}, {0., 0., -1.}},
	{{0.5, -0.809017, 0.309017}, {0.809017, 0.309017, -0.5}, {0.309017, 0.5, 0.809017}},
	{{-0.5, 0.809017, 0.309017}, {-0.809017, -0.309017, -0.5}, {-0.309017, -0.5, 0.809017}},
	{{-0.809017, 0.309017, 0.5}, {0.309017, -0.5, 0.809017}, {0.5, 0.809017, 0.309017}},
	{{0.309017, 0.5, -0.809017}, {-0.5, 0.809017, 0.309017}, {0.809017, 0.309017, 0.5}},
	{{0.309017, -0.5, 0.809017}, {-0.5, -0.809017, -0.309017}, {0.809017, -0.309017, -0.5}},
	{{0.809017, -0.309017, 0.5}, {-0.309017, 0.5, 0.809017}, {-0.5, -0.809017, 0.309017}},
	{{-0.309017, 0.5, 0.809017}, {0.5, 0.809017, -0.309017}, {-0.809017, 0.309017, -0.5}},
	{{0.809017, 0.309017, -0.5}, {-0.309017, -0.5, -0.809017}, {-0.5, 0.809017, -0.309017}},
	{{0.5, 0.809017, -0.309017}, {0.809017, -0.309017, 0.5}, {0.309017, -0.5, -0.809017}},
	{{-0.809017, -0.309017, -0.5}, {0.309017, 0.5, -0.809017}, {0.5, -0.809017, -0.309017}},
	{{-0.309017, -0.5, -0.809017}, {0.5, -0.809017, 0.309017}, {-0.809017, -0.309017, 0.5}},
	{{-0.5, -0.809017, -0.309017}, {-0.809017, 0.309017, 0.5}, {-0.309017, 0.5, -0.809017}},
	{{-0.309017, -0.5, 0.809017}, {0.5, -0.809017, -0.309017}, {0.809017, 0.309017, 0.5}},
	{{0.309017, 0.5, 0.809017}, {-0.5, 0.809017, -0.309017}, {-0.809017, -0.309017, 0.5}},
	{{-0.5, 0.809017, -0.309017}, {-0.809017, -0.309017, 0.5}, {0.309017, 0.5, 0.809017}},
	{{0.809017, -0.309017, -0.5}, {-0.309017, 0.5, -0.809017}, {0.5, 0.809017, 0.309017}},
	{{0.809017, 0.309017, 0.5}, {-0.309017, -0.5, 0.809017}, {0.5, -0.809017, -0.309017}},
	{{0.5, -0.809017, -0.309017}, {0.809017, 0.309017, 0.5}, {-0.309017, -0.5, 0.809017}},
	{{-0.809017, -0.309017, 0.5}, {0.309017, 0.5, 0.809017}, {-0.5, 0.809017, -0.309017}},
	{{0.5, 0.809017, 0.309017}, {0.809017, -0.309017, -0.5}, {-0.309017, 0.5, -0.809017}},
	{{-0.309017, 0.5, -0.809017}, {0.5, 0.809017, 0.309017}, {0.809017, -0.309017, -0.5}},
	{{-0.5, -0.809017, 0.309017}, {-0.809017, 0.309017, -0.5}, {0.309017, -0.5, -0.809017}},
	{{-0.809017, 0.309017, -0.5}, {0.309017, -0.5, -0.809017}, {-0.5, -0.809017, 0.309017}},
	{{0.309017, -0.5, -0.809017}, {-0.5, -0.809017, 0.309017}, {-0.809017, 0.309017, -0.5}},
	{{-0.309017, 0.5, 0.809017}, {-0.5, -0.809017, 0.309017}, {0.809017, -0.309017, 0.5}},
	{{0.309017, -0.5, 0.809017}, {0.5, 0.809017, 0.309017}, {-0.809017, 0.309017, 0.5}},
	{{0.5, 0.809017, -0.309017}, {-0.809017, 0.309017, -0.5}, {-0.309017, 0.5, 0.809017}},
	{{0.809017, -0.309017, 0.5}, {0.309017, -0.5, -0.809017}, {0.5, 0.809017, -0.309017}},
	{{0.809017, 0.309017, -0.5}, {0.309017, 0.5, 0.809017}, {0.5, -0.809017, 0.309017}},
	{{-0.5, -0.809017, -0.309017}, {0.809017, -0.309017, -0.5}, {0.309017, -0.5,   0.809017}},
	{{-0.809017, -0.309017, -0.5}, {-0.309017, -0.5,   0.809017}, {-0.5, 0.809017, 0.309017}},
	{{-0.5, 0.809017,   0.309017}, {0.809017, 0.309017, 0.5}, {0.309017,   0.5, -0.809017}},
	{{-0.309017, -0.5, -0.809017}, {-0.5,   0.809017, -0.309017}, {0.809017, 0.309017, -0.5}},
	{{0.5, -0.809017,   0.309017}, {-0.809017, -0.309017,   0.5}, {-0.309017, -0.5, -0.809017}},
	{{-0.809017, 0.309017,   0.5}, {-0.309017,   0.5, -0.809017}, {-0.5, -0.809017, -0.309017}},
	{{0.309017,   0.5, -0.809017}, {0.5, -0.809017, -0.309017}, {-0.809017, -0.309017, -0.5}},
	{{0.5, 0.809017, 0.309017}, {-0.809017, 0.309017,   0.5}, {0.309017, -0.5, 0.809017}},
	{{-0.5, -0.809017,   0.309017}, {0.809017, -0.309017, 0.5}, {-0.309017, 0.5,   0.809017}},
	{{0.809017, 0.309017, 0.5}, {0.309017,   0.5, -0.809017}, {-0.5, 0.809017, 0.309017}},
	{{0.309017, 0.5,   0.809017}, {0.5, -0.809017, 0.309017}, {0.809017,   0.309017, -0.5}},
	{{0.309017, -0.5, -0.809017}, {0.5,   0.809017, -0.309017}, {0.809017, -0.309017,   0.5}},
	{{-0.809017, -0.309017, 0.5}, {-0.309017, -0.5, -0.809017}, {0.5, -0.809017, 0.309017}},
	{{-0.309017, 0.5, -0.809017}, {-0.5, -0.809017, -0.309017}, {-0.809017, 0.309017, 0.5}},
	{{-0.809017, 0.309017, -0.5}, {-0.309017, 0.5, 0.809017}, {0.5, 0.809017, -0.309017}},
	{{0.5, -0.809017, -0.309017}, {-0.809017, -0.309017, -0.5}, {0.309017, 0.5, -0.809017}},
	{{0.809017, -0.309017, -0.5}, {0.309017, -0.5, 0.809017}, {-0.5, -0.809017, -0.309017}},
	{{-0.309017, -0.5, 0.809017}, {-0.5, 0.809017, 0.309017}, {-0.809017, -0.309017, -0.5}},
	{{-0.5, 0.809017, -0.309017}, {0.809017, 0.309017, -0.5}, {-0.309017, -0.5, -0.809017}}
} ;

/* Icosahedral symmetrization
 * 	Assumes vertices are at permutations of (0, +-1, +-tau)
 * 	Arguments:
 * 		Pointer to model representing centered 3D volume
 * 		Size of model
 * 	No return value. Symmetrization performed in-place
 */
void symmetrize_icosahedral(double *model, int size) {
	double *temp = malloc(size*size*size*sizeof(double)) ;
	int i ;
	
	memcpy(temp, model, size*size*size*sizeof(double)) ;
	memset(model, 0, size*size*size*sizeof(double)) ;
	
	for (i = 0 ; i < 60 ; ++i)
		rotate_model(icos_list[i], temp, size, model) ;
	
	for (i = 0 ; i < size*size*size ; ++i)
		model[i] /= 60. ;
}
