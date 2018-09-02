/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double sigma_pos[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 100;
	normal_distribution<double> dist_x(x, sigma_pos[0]);
	normal_distribution<double> dist_y(y, sigma_pos[1]);
	normal_distribution<double> dist_theta(theta, sigma_pos[2]);
	for(unsigned int i = 0; i < num_particles; i++){
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(1.0);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for(unsigned int i = 0; i < num_particles; i++){
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		if (fabs(yaw_rate) < 0.00001) {  
      		x += velocity * delta_t * cos(theta);
      		y += velocity * delta_t * sin(theta);
    	} else{
			x += velocity/yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			y += velocity/yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
		}
		theta = theta + yaw_rate*delta_t;
		particles[i].x = x + dist_x(gen);
		particles[i].y = y + dist_y(gen);
		particles[i].theta = theta + dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(unsigned int o=0; o < observations.size(); o++){
		int closest_id = -1;
		float closest_dist = 10000000.0;
		for(unsigned int p=0; p < predicted.size(); p++){
			double op_dist = dist(observations[o].x, observations[o].y, predicted[p].x, predicted[p].y);
			if(op_dist < closest_dist){
				closest_dist = op_dist;
				closest_id = predicted[p].id;
			}
		}
		observations[o].id = closest_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// I pre-set the size of the observations vector before the loop so that I don't use push_back
    std::vector<LandmarkObs> trans_observations;
    trans_observations.resize(observations.size());

	for(unsigned int i=0; i < num_particles; i++){
		Particle p = particles[i];
		// Transform the coords of observed landmark from car's to world's
		for(unsigned int o=0; o < observations.size(); o++){
			double xm = p.x + observations[o].x * cos(p.theta) - observations[o].y * sin(p.theta);
			double ym = p.y + observations[o].x * sin(p.theta) + observations[o].y * cos(p.theta);
			trans_observations[o] = {observations[o].id, xm, ym};
		}
		// Only consider landmarks that are within the range of the sensor
		std::vector<LandmarkObs> predicted_landmarks;
		for(unsigned int l=0; l < map_landmarks.landmark_list.size();l++){
			LandmarkObs cand_lm;
			cand_lm.x = map_landmarks.landmark_list[l].x_f;
			cand_lm.y = map_landmarks.landmark_list[l].y_f;
			cand_lm.id = map_landmarks.landmark_list[l].id_i;
			if(sensor_range >= dist(p.x, p.y, cand_lm.x, cand_lm.y)){
				predicted_landmarks.push_back(cand_lm);
			}
		}
		// Associate each observed landmark to the nearest landmark on the map
		// trans_observations[{id_1, x_1, y_1}, {id_2, x_2, y_2}, {id_3, x_3, y_3}]
		dataAssociation(predicted_landmarks, trans_observations);

		double prob = 1.0;
		for(unsigned int o=0; o < trans_observations.size(); o++){
			for(unsigned int l=0; l < predicted_landmarks.size(); l++){
				LandmarkObs lm_l = predicted_landmarks[l];
				if(trans_observations[o].id == lm_l.id){
					prob *= mv_gaussian(lm_l.x, std_landmark[0], trans_observations[o].x, 
										lm_l.y, std_landmark[1], trans_observations[o].y);
					break;
				}
			}
		}
		particles[i].weight = prob;
		weights[i] = prob;
	}
	//
}



void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> d(weights.begin(), weights.end());
	std::vector<Particle> res_particles;
	res_particles.resize(num_particles);
	std::vector<double> res_weights;
	res_weights.resize(num_particles);

	for(unsigned int i=0; i < num_particles; i++){
		int p_indx = d(gen);
		res_particles[i] = particles[p_indx];
		res_weights[i] = weights[p_indx];
	}
	particles = res_particles;
	weights = res_weights;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
