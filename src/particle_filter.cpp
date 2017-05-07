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
#include <cfloat>
#include <limits>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 1000;

    std::default_random_engine g;
    std::normal_distribution<double> gen_x(x, std[0]);
    std::normal_distribution<double> gen_y(y, std[1]);
    std::normal_distribution<double> gen_theta(theta, std[2]);

    for(int i = 0; i < num_particles; ++i) {
        const double weight = 1;
        particles.push_back({i, gen_x(g), gen_y(g), gen_theta(g), weight});
        weights.push_back(1);
    }
    is_initialized = true;
}

void update_bicycle(double & x, double & y, double & theta, double velocity, double yaw_rate, double dt) {
    // see https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/2c318113-724b-4f9f-860c-cb334e6e4ad7/lessons/5c50790c-5370-4c80-aff6-334659d5c0d9/concepts/ca98c146-ee0d-4e53-9900-81cec2b771f7
    //x = x + (velocity / yaw_rate) * (sin(theta + yaw_rate * dt) - sin(theta));
    //y = y + (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * dt));
    x += velocity * dt * cos(theta);
    y += velocity * dt * sin(theta);
    theta = theta + yaw_rate * dt;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
    // standard deviation of yaw [rad]]


    std::normal_distribution<double> noise_x(0, std_pos[0]);
    std::normal_distribution<double> noise_y(0, std_pos[1]);
    std::normal_distribution<double> noise_yaw(0, std_pos[2]);
    std::default_random_engine generator;

    for(Particle& particle : particles) {
        update_bicycle(particle.x, particle.y, particle.theta, velocity, yaw_rate, delta_t);

        // add a little bit of noise
        particle.x += noise_x(generator);
        particle.y += noise_y(generator);
        particle.theta += noise_yaw(generator);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations, Map& map) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for(LandmarkObs & observation : observations) {
        // find the lowest distance squared
        double best_d2 = DBL_MAX;
        int best_i = -1;
        for(int i = 0; i < map.landmark_list.size(); ++i) {
            auto landmark = map.landmark_list.at(i);
            double dx = observation.x - landmark.x_f;
            double dy = observation.y - landmark.y_f;
            double d2 = dx*dx + dy*dy;
            if(d2 < best_d2) {
                best_i = i;
                best_d2 = d2;
            }
        }
        Map::single_landmark_s best_map = map.landmark_list[best_i];
        LandmarkObs best_observation = {best_map.id_i, best_map.x_f, best_map.y_f};
        predicted.push_back(best_observation);
    }

}

double normal_density(double dx, double variance) {
    return exp(- (dx*dx / (variance) / 2.0) / sqrt(2.0 * M_PI * variance));
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

    for(Particle & particle : particles) {

        // change observations from car to world coordinates
        std::vector<LandmarkObs> world_observations;
        for(LandmarkObs & car_observation : observations) {
            LandmarkObs world_observation;
            world_observation.id = car_observation.id;
            double alpha = particle.theta;
            world_observation.x = particle.x + car_observation.x * cos(alpha) - car_observation.y * sin(alpha);
            world_observation.y = particle.y + car_observation.x * sin(alpha) + car_observation.y * cos(alpha);
            // positive is down on the map
//world_observation.y = -world_observation.y;
            world_observations.push_back(world_observation);
        }

        // find associated points
        std::vector<LandmarkObs> associations;
        dataAssociation(associations, world_observations, map_landmarks);

        // update weight with probability
        double p = 1.0;
        for(int i=0; i < associations.size(); i++) {
            const double variance = 4.0;
            double d = dist(associations[i].x, associations[i].y, world_observations[i].x, world_observations[i].y);
            p *= normal_density(d, variance);
        }
        particle.weight = p;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<double> weights;
    for(Particle p : particles) {
        weights.push_back(p.weight);
    }
    std::discrete_distribution<> distribution(weights.begin(), weights.end());
    std::vector<Particle> new_particles;
    std::default_random_engine generator;
    for(int i = 0; i < particles.size(); i++) {
        new_particles.push_back(particles[distribution(generator)]);
    }
    particles = new_particles;


}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
