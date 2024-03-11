#include "params.h"

void params_from_config(char *config_fname, char *config_section, struct params *param) {
	char line[2048], section_name[1024], config_folder[1024], temp[8] ;
	char beta_str[1024] = {'\0'} ;
	char *temp_fname = strndup(config_fname, 1024) ;
	sprintf(config_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	
	param->known_scale = 0 ;
	param->start_iter = 1 ;
	param->beta_period = 100 ;
	param->beta_jump = 1. ;
	param->beta_factor = 1. ;
	param->radius = 0. ;
	param->radius_period = 100 ;
	param->radius_jump = 0. ;
	param->oversampling = 10. ;
	param->need_scaling = 0 ;
	param->update_scale = 1 ;
	param->alpha = 0. ;
	param->beta = NULL ;
	param->beta_start = malloc(1 * sizeof(double)) ;
	param->beta_start[0] = -1. ;
	param->sigmasq = 0. ;
	param->modes = 1 ;
	param->nonrot_modes = 0 ;
	param->rot_per_mode = 0 ;
	param->recon_type = RECON3D ;
	param->friedel_sym = 0 ;
	param->axial_sym = 1 ;
	param->save_prob = 0 ;
	param->refine = 0 ;
	param->coarse_div = 0 ;
	param->fine_div = 0 ;
	sprintf(param->log_fname, "%.1015s/EMC.log", config_folder) ;
	sprintf(param->output_folder, "%.1017s/data/", config_folder) ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 2048, config_fp) != NULL) {
		char *token ;
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, config_section) == 0) {
			if (strcmp(token, "output_folder") == 0)
				absolute_strcpy(config_folder, param->output_folder, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "log_file") == 0)
				absolute_strcpy(config_folder, param->log_fname, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "recon_type") == 0) {
				strncpy(temp, strtok(NULL, " =\n"), 8) ;
				if (strcmp(temp, "3d") == 0)
					param->recon_type = RECON3D ;
				else if (strcmp(temp, "2d") == 0)
					param->recon_type = RECON2D ;
				else if (strcmp(temp, "rz") == 0)
					param->recon_type = RECONRZ ;
				else
					fprintf(stderr, "WARNING! Unknown recon_type %s. Assuming 3D reconstruction.\n", temp) ;
			}
			else if (strcmp(token, "need_scaling") == 0)
				param->need_scaling = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "alpha") == 0)
				param->alpha = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "beta") == 0)
				strcpy(beta_str, strtok(NULL, " =\n")) ;
			else if (strcmp(token, "beta_factor") == 0)
				param->beta_factor = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "radius") == 0)
				param->radius = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "num_modes") == 0)
				param->modes = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "num_nonrot_modes") == 0)
				param->nonrot_modes = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "num_rot") == 0)
				param->rot_per_mode = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "beta_schedule") == 0) {
				param->beta_jump = atof(strtok(NULL, " =\n")) ;
				param->beta_period = atoi(strtok(NULL, " =\n")) ;
			}
			else if (strcmp(token, "radius_schedule") == 0) {
				param->radius_jump = atof(strtok(NULL, " =\n")) ;
				param->radius_period = atoi(strtok(NULL, " =\n")) ;
			}
			else if (strcmp(token, "oversampling") == 0)
				param->oversampling = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "gaussian_sigma") == 0) {
				param->sigmasq = atof(strtok(NULL, " =\n")) ;
				param->sigmasq *= param->sigmasq ;
				fprintf(stderr, "sigma_squared = %f\n", param->sigmasq) ;
			}
			else if (strcmp(token, "friedel_sym") == 0)
				param->friedel_sym = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "axial_sym") == 0)
				param->axial_sym = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "save_prob") == 0)
				param->save_prob = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "update_scale") == 0)
				param->update_scale = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "num_div") == 0) {
				param->fine_div = atoi(strtok(NULL, " =\n")) ;
				char *cstr = strtok(NULL, " =\n") ;
				if (cstr == NULL) {
					param->fine_div = 0 ;
				}
				else {
					param->refine = 1; 
					param->coarse_div = atoi(cstr) ;
					fprintf(stderr, "Doing refinement from num_div = %d -> %d\n", param->coarse_div, param->fine_div) ;
				}
			}
		}
	}
	fclose(config_fp) ;
	
	if (strcmp(beta_str, "auto") == 0)
		param->beta_start[0] = -1. ;
	else if (beta_str[0] != '\0')
		param->beta_start[0] = atof(beta_str) ;
	
	if (!param->rank)
		fprintf(stderr, "Parsed params from config file\n") ;
}

void generate_output_dirs(struct params *param) {
	char line[2048] ;
	
	sprintf(line, "%s/output", param->output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/weights", param->output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/mutualInfo", param->output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/scale", param->output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/orientations", param->output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/likelihood", param->output_folder) ;
	mkdir(line, 0750) ;
	sprintf(line, "%s/probabilities", param->output_folder) ;
	mkdir(line, 0750) ;
	if (param->modes > 1) {
		sprintf(line, "%s/modes", param->output_folder) ;
		mkdir(line, 0750) ;
	}
}

void free_params(struct params *param) {
	if (param->beta != NULL)
		free(param->beta) ;
	free(param->beta_start) ;
	free(param) ;
}
