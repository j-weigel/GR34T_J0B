# Boids: Swarming as Emergence (and Evolving It)

DCS 340 · Project II · [John Weigel]

This notebook explores the concept of swarming, Reynolds (1987). My personal interest in the topic comes from a cognitive and behavioral neuroscience approach. Convergent evolution has reproduced this complex adaptive system many times using simple rules. Reynolds computationally models this behavior in individual boids. 

First, the model is introduced. There are three local rules that each bird-oid (boid) must follow in priority order: separation (or steering away from overcrowding), alignment (matching the average velocity of neighbors), and cohesion (steering toward the average position of nearby flockmates). Each boid also has a defined radius of perception, in which only those immediate neighbors influence their behavior (not the global flock). 

The notebook also quantifies flock behavior. Measuring emergent behavior requires defining the behavior: polarization (Vicsek et al.,1995; 0 = chaos, 1 = complete alignment), nearest neighbor distance, and total number of groups all help quantify when flocking behavior emerges. 

Reynolds tunes the model himself, as the notebook allows for as well. What if we wanted a genetic algorithm to tune the model? The notebook does this next. Using the three rules and perception radius as changing genome weights and the emergent behavior measurements established earlier as the fitness function, a genetic algorithm searches for the best settings to produce flocking behavior. High polarization + most boids in one group − collisions, averaged over the whole run = fitness score. Flocks formed earlier score higher. 

Class activities are available to help learn the concept more/explore the model. Potential extensions also include adding stagnant objects or predator/s that the boids must navigate. 
