#ifndef QUAT_H
#define QUAT_H

struct rotation {
	int num_rot, num_rot_p ;
	double *quat ;
	int icosahedral_flag ;
} ;

int quat_gen(int, struct rotation*) ;
int parse_quat(char*, struct rotation*) ;
void divide_quat(int, int, struct rotation*) ;

#endif //QUAT_H
