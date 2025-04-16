import asyncio
import platform
import pygame
import numpy as np
import random
import pandas as pd
from copy import deepcopy

# Neural Network Class
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            for i in range(len(layer_sizes)-1)
        ]
        self.biases = [
            np.random.randn(1, layer_sizes[i+1]) * 0.1
            for i in range(len(layer_sizes)-1)
        ]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, inputs):
        current = np.array(inputs)
        for w, b in zip(self.weights, self.biases):
            current = self.sigmoid(np.dot(current, w) + b)
        return current[0]

# Bird Class
class Bird:
    def __init__(self, nn_layers=[4, 8, 1]):
        self.y = 300
        self.velocity = 0
        self.gravity = 0.8
        self.jump = -10
        self.score = 0
        self.fitness = 0
        self.nn = NeuralNetwork(nn_layers)
        self.alive = True

    def think(self, inputs):
        output = self.nn.forward(inputs)
        if output > 0.5:
            self.velocity = self.jump

    def update(self):
        if self.alive:
            self.velocity += self.gravity
            self.y += self.velocity
            self.score += 1
            if self.y > 600 or self.y < 0:
                self.alive = False

# Pipe Class
class Pipe:
    def __init__(self):
        self.x = 800
        self.gap = 200
        self.height = random.randint(150, 450)
        self.passed = False

    def update(self):
        self.x -= 5

    def collides(self, bird):
        bird_rect = pygame.Rect(100, bird.y - 20, 40, 40)
        top_pipe = pygame.Rect(self.x, 0, 50, self.height)
        bottom_pipe = pygame.Rect(self.x, self.height + self.gap, 50, 600)
        return bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe)

# Genetic Algorithm Functions
def custom_crossover(parent1, parent2, crossover_rate=0.7):
    child = deepcopy(parent1)
    for i in range(len(child.nn.weights)):
        if random.random() < crossover_rate:
            # Blend weights and biases
            alpha = random.random()
            child.nn.weights[i] = alpha * parent1.nn.weights[i] + (1 - alpha) * parent2.nn.weights[i]
            child.nn.biases[i] = alpha * parent1.nn.biases[i] + (1 - alpha) * parent2.nn.biases[i]
        # Add small mutation
        if random.random() < 0.1:
            child.nn.weights[i] += np.random.randn(*child.nn.weights[i].shape) * 0.05
            child.nn.biases[i] += np.random.randn(*child.nn.biases[i].shape) * 0.05
    return child

def select_parent(population, fitnesses):
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, bird in enumerate(population):
        current += fitnesses[i]
        if current > pick:
            return bird
    return population[-1]

# Game Setup
FPS = 60
POPULATION_SIZE = 50
birds = [Bird() for _ in range(POPULATION_SIZE)]
pipes = [Pipe()]
generation = 1
best_fitness = 0
best_bird = None

def setup():
    global screen, clock
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Flappy Bird AI")
    clock = pygame.time.Clock()

def update_loop():
    global birds, pipes, generation, best_fitness, best_bird

    # Update pipes
    if len(pipes) == 0 or pipes[-1].x < 500:
        pipes.append(Pipe())
    pipes = [pipe for pipe in pipes if pipe.x > -50]
    for pipe in pipes:
        pipe.update()

    # Update birds
    alive_birds = sum(1 for bird in birds if bird.alive)
    if alive_birds == 0:
        # Create new generation
        fitnesses = [bird.fitness for bird in birds]
        if max(fitnesses) > best_fitness:
            best_fitness = max(fitnesses)
            best_bird = deepcopy(birds[np.argmax(fitnesses)])
        
        new_birds = []
        for _ in range(POPULATION_SIZE):
            parent1 = select_parent(birds, fitnesses)
            parent2 = select_parent(birds, fitnesses)
            child = custom_crossover(parent1, parent2)
            new_birds.append(child)
        birds = new_birds
        pipes = [Pipe()]
        generation += 1

    for bird in birds:
        if not bird.alive:
            continue

        # Neural network inputs
        next_pipe = next((pipe for pipe in pipes if pipe.x + 50 > 100), pipes[0])
        inputs = [
            bird.y / 600,  # Normalized bird y position
            bird.velocity / 10,  # Normalized velocity
            next_pipe.x / 800,  # Normalized pipe x position
            (next_pipe.height + next_pipe.gap/2) / 600  # Normalized pipe gap center
        ]
        bird.think(inputs)
        bird.update()

        # Check collisions
        for pipe in pipes:
            if pipe.collides(bird):
                bird.alive = False
                bird.fitness = bird.score
                break

        # Update fitness for passing pipe
        if next_pipe.x + 50 < 100 and not next_pipe.passed:
            bird.score += 100
            next_pipe.passed = True

    # Render
    screen.fill((135, 206, 235))  # Sky blue
    for pipe in pipes:
        pygame.draw.rect(screen, (0, 255, 0), (pipe.x, 0, 50, pipe.height))
        pygame.draw.rect(screen, (0, 255, 0), (pipe.x, pipe.height + pipe.gap, 50, 600))
    for bird in birds:
        if bird.alive:
            pygame.draw.circle(screen, (255, 255, 0), (100, int(bird.y)), 20)
    
    # Display info
    font = pygame.font.Font(None, 36)
    text = font.render(f"Gen: {generation} Alive: {alive_birds} Best: {best_fitness}", True, (0, 0, 0))
    screen.blit(text, (10, 10))
    
    pygame.display.flip()
    clock.tick(FPS)

async def main():
    setup()
    while True:
        update_loop()
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())