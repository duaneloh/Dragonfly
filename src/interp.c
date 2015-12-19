#include "emc.h"

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
				rot_pix[i] += rot[i][j] * detector[t*3 + j] ;
			rot_pix[i] += center ;
		}
		
		tx = rot_pix[0] ;
		ty = rot_pix[1] ;
		tz = rot_pix[2] ;
		
		if (tx < 0 || tx > size-2 || ty < 0 || ty > size-2 || tz < 0 || tz > size-2) {
			slice[t] = 1.e-10 ;
			continue ;
		}
		
		x = (int) tx ;
		y = (int) ty ;
		z = (int) tz ;
		fx = tx - x ;
		fy = ty - y ;
		fz = tz - z ;
		cx = 1. - fx ;
		cy = 1. - fy ;
		cz = 1. - fz ;
		
		slice[t] =	cx*cy*cz*model3d[x*size*size + y*size + z] +
				cx*cy*fz*model3d[x*size*size + y*size + ((z+1)%size)] +
				cx*fy*cz*model3d[x*size*size + ((y+1)%size)*size + z] +
				cx*fy*fz*model3d[x*size*size + ((y+1)%size)*size + ((z+1)%size)] +
				fx*cy*cz*model3d[((x+1)%size)*size*size + y*size + z] +
				fx*cy*fz*model3d[((x+1)%size)*size*size + y*size + ((z+1)%size)] + 
				fx*fy*cz*model3d[((x+1)%size)*size*size + ((y+1)%size)*size + z] + 
				fx*fy*fz*model3d[((x+1)%size)*size*size + ((y+1)%size)*size + ((z+1)%size)] ;
		
		// Correct for solid angle and polarization
		slice[t] *= pow(1. + detector[t*3 + 2] / detd, 3.) * (1. - pow(detector[t*3 + 1]/detd, 2.)) ;
		
		// Use rescale as flag on whether to take log or not
		if (slice[t] <= 0.)
//			slice[t] = DBL_MIN ;
			slice[t] = 1.e-10 ;
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
				rot_pix[i] += rot[i][j] * detector[t*3 + j] ;
			rot_pix[i] += center ;
		}
		
		tx = rot_pix[0] ;
		ty = rot_pix[1] ;
		tz = rot_pix[2] ;
		
		if (tx < 0 || tx > size-2 || ty < 0 || ty > size-2 || tz < 0 || tz > size-2)
			continue ;
		
		x = tx ;
		y = ty ;
		z = tz ;
		fx = tx - x ;
		fy = ty - y ;
		fz = tz - z ;
		cx = 1. - fx ;
		cy = 1. - fy ;
		cz = 1. - fz ;
		
		// Correct for solid angle and polarization
		slice[t] /= pow(1. + detector[t*3 + 2] / detd, 3.) * (1. - pow(detector[t*3 + 1]/detd, 2.)) ;
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
