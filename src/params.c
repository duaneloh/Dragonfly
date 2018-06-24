#include "params.h"

static char *generate_token(char *line, char *section_name) {
	char *token = strtok(line, " =") ;
	if (token[0] == '#' || token[0] == '\n')
		return NULL ;
	
	if (line[0] == '[') {
		token = strtok(line, "[]") ;
		strcpy(section_name, token) ;
		return NULL ;
	}
	
	return token ;
}

static void absolute_strcpy(char *config_folder, char *path, char *rel_path) {
	if (rel_path[0] == '/' || strstr(rel_path, ":::") != NULL) {
		strcpy(path, rel_path) ;
	}
	else {
		strcpy(&path[strlen(config_folder)], rel_path) ;
		strncpy(path, config_folder, strlen(config_folder)) ;
	}
}

void generate_params(char *config_fname, struct params *param) {
	//char line[1024], section_name[1024], config_folder[1024], *token ;
	char line[1024], section_name[1024], config_folder[1024], temp[8] ;
	char *temp_fname = strndup(config_fname, 1024) ;
	sprintf(config_folder, "%s/", dirname(temp_fname)) ;
	free(temp_fname) ;
	
	param->known_scale = 0 ;
	param->start_iter = 1 ;
	param->beta_period = 100 ;
	param->beta_jump = 1. ;
	param->need_scaling = 0 ;
	param->alpha = 0. ;
	param->beta = 1. ;
	param->sigmasq = 0. ;
	param->modes = 1 ;
	param->rot_per_mode = 0 ;
	param->recon_type = RECON3D ;
	sprintf(param->log_fname, "%s/EMC.log", config_folder) ;
	sprintf(param->output_folder, "%s/data/", config_folder) ;
	
	FILE *config_fp = fopen(config_fname, "r") ;
	while (fgets(line, 1024, config_fp) != NULL) {
		char *token ;
		if ((token = generate_token(line, section_name)) == NULL)
			continue ;
		
		if (strcmp(section_name, "emc") == 0) {
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
				else
					fprintf(stderr, "WARNING! Unknown recon_type %s. Assuming 3D reconstruction.\n", temp) ;
			}
			else if (strcmp(token, "need_scaling") == 0)
				param->need_scaling = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "alpha") == 0)
				param->alpha = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "beta") == 0)
				param->beta = atof(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "num_modes") == 0)
				param->modes = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "num_rot") == 0)
				param->rot_per_mode = atoi(strtok(NULL, " =\n")) ;
			else if (strcmp(token, "beta_schedule") == 0) {
				param->beta_jump = atof(strtok(NULL, " =\n")) ;
				param->beta_period = atoi(strtok(NULL, " =\n")) ;
			}
			else if (strcmp(token, "gaussian_sigma") == 0) {
				param->sigmasq = atof(strtok(NULL, " =\n")) ;
				param->sigmasq *= param->sigmasq ;
				fprintf(stderr, "sigma_squared = %f\n", param->sigmasq) ;
			}
		}
	}
	fclose(config_fp) ;
	if (!param->rank)
		fprintf(stderr, "Parsed params from config file\n") ;
}

void generate_output_dirs(struct params *param) {
	char line[1024] ;
	
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
}

