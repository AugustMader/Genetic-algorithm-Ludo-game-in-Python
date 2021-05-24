import unittest
import sys
import numpy as np
import pandas as pd
from numpy.lib.function_base import append
sys.path.append("../")

class geneticAlgorithm:
    def __init__(self, popsize = 100):
        self.populationSize = popsize
        self.utilitySize = 12
        self.succesCriteria = 0.85 # 95% win rate
        self.bestScoreOverall = 0
        self.trainingGames = 100
        self.eliteProcent = 0.10 # bestprocent
        self.elitePopulations = int(self.populationSize*self.eliteProcent)
        self.mutationSTD = 0.25

    def initPopulation(self):
        self.utilityFunction = []
        for x in range(self.populationSize):
            rand = np.random.randint(0,100 + 1,size=self.utilitySize)
            self.utilityFunction.append(rand)

    def checkGlobe(self, newPosition):
        globe = [9, 14, 22, 27, 35, 40, 48, 53]
        
        if (newPosition in globe):
            return True
        return False

    def checkDangerous(self, newPosition, enemy_pieces):
        danger = [14, 27, 40]

        # Check if globe is dangerous
        for x in range(len(danger)):
            if newPosition == danger[x]:
                if 0 in enemy_pieces[x]:
                    return True
        
        # check if opponent can hit player home
        danger = []
        for x in range(5):
            if  0 < newPosition - (x + 1) :
                danger.append(newPosition - (x + 1))
            else:
                danger.append(53 - (x + 1))
    
        #print("Danger:",danger)
        #print("Ememy pieces", enemy_pieces)
        for x in range(len(enemy_pieces)):
            for i in range(len(enemy_pieces[x])):
                if enemy_pieces[x][i] in danger:
                    return True
        
        return False
    

    def checkStar(self, newPosition):
        star = [5, 12, 18, 25, 31, 38, 44, 51]
        if (newPosition in star):
            return True
        return False


    def enterGoalStraight(self, newPosition):
        if 53 < newPosition:
            return True
        return False

    def checkHitOpponent(self, newPosition, enemy_pieces):
        for x in range(len(enemy_pieces)):
            for i in range(len(enemy_pieces[x])):
                if newPosition == enemy_pieces[x][i]:
                    return True
        return False

    def checkHitHomeItself(self, newPosition, enemy_pieces):
        globe = [9, 14, 22, 27, 35, 40, 48, 53]
        
        if (newPosition in globe):
            for x in range(len(enemy_pieces)):
                for i in range(len(enemy_pieces[x])):
                    if(enemy_pieces[x][i] == newPosition):
                        return True
        return False
    
    def escapePossibleBeat(self, brickPosition, enemy_pieces):
        globe = [9, 14, 22, 27, 35, 40, 48, 53]
        
        if brickPosition in globe:
            return False
        
        danger = []
        for x in range(5):
            if  0 < brickPosition - (x + 1) :
                danger.append(brickPosition - (x + 1))
            else:
                danger.append(53 - (x + 1))
        
        #print(danger)
        for x in range(len(enemy_pieces)):
            for i in range(len(enemy_pieces[x])):
                if enemy_pieces[x][i] in danger:
                    return True
        
        return False

    def checkGoalReached(self, newPosition):
        if newPosition == 59 or newPosition == 51:
            return True
        return False

    def checkStandingOnGlobe(self, brickPosition):
        globe = [1, 9, 14, 22, 27, 35, 40, 48, 53]
        if brickPosition in globe:
            return True
        return False

    def checkPossibleBeatOpponent(self, newPosition, enemy_pieces):
        hit = []

        for x in range(6):
            if newPosition + (x + 1) < 52:
                hit.append(newPosition + (x + 1))
        #print(hit)
        for x in range(len(enemy_pieces)):
            for i in range(len(enemy_pieces[x])):
                if enemy_pieces[x][i] in hit:
                    return True

        return False

    def adjustEnemyPieces(self, enemy_pieces):
        offset = [13, 26, 39]

        for x in range(len(enemy_pieces)):
            for i in range(len(enemy_pieces[x])):
                if not (enemy_pieces[x][i] == 59 or enemy_pieces[x][i] == 0):
                    if 53 < enemy_pieces[x][i]:
                        enemy_pieces[x][i] = 60
                    else:
                        enemy_pieces[x][i] += offset[x]
                        enemy_pieces[x][i] = enemy_pieces[x][i] % 52
        return enemy_pieces

    def determineStateSpace(self, brickPosition, dice, enemy_pieces):
        states = []
        #print(enemy_pieces)
        #enemy_pieces = [[11,  7,  0,  1],
        #                [ 0,  0,  0, 25],
        #                [24,  0,  0,  0]]

        # Check if brick is on the board
        if brickPosition != 0:
            newPosition = brickPosition + dice
            #newPosition = 23

            #print("newPosition",newPosition)

            # 0 - Hit opponent home
            if self.checkHitOpponent(newPosition, enemy_pieces):
                states.append(0)

            # 1 - Brick can enter the goal straigt
            if self.enterGoalStraight(newPosition):
                states.append(1)

            # 2 - Brick can hit itself home
            if self.checkHitHomeItself(newPosition, enemy_pieces):
                states.append(2)

            # 3 - Brick can hit on a star
            if self.checkStar(newPosition):
                states.append(3)

            # 5 - Hit globe
            if 2 not in states:
                if self.checkGlobe(newPosition):
                    states.append(5)

            # 4 - Enter dangerous field
            if 3 not in states:
                if 5 not in states:
                    if self.checkDangerous(newPosition, enemy_pieces):
                        states.append(4)

            # 8 - Get into goal
            if self.checkGoalReached(newPosition):
                states.append(8)

            # 9 - Escape possible beat
            if self.escapePossibleBeat(brickPosition,enemy_pieces):
                states.append(9)

            # 10 - Standing on globe
            if self.checkStandingOnGlobe(brickPosition):
                states.append(10)

            if self.checkPossibleBeatOpponent(newPosition, enemy_pieces):
                states.append(11)

        # 6 - Move out of start
        if brickPosition == 0 and dice == 6:
            states.append(6)

        # 7 - State not described // Default case
        if len(states) == 0:
            states.append(7)

        #print(states)
        
        #if 2 in states:
        #    states = []
        #    states.append(2)

        return states
        #print(states)

    def selection(self, score):
        print("selection")
        scoreSorted = score.copy()
        scoreSorted.sort(reverse = True)
        elite = []
        for x in range(self.elitePopulations):
            highest = scoreSorted[x]
            index = score.index(highest)
            scoreSorted[x] = 0
            score[index] = 0
            e = self.utilityFunction[index]
            elite.append(e)
        self.utilityFunction = []
        self.utilityFunction = elite
        return elite
        
    def crossover(self, elite):
        print("crossover")
       
        for x in range(self.populationSize - self.elitePopulations):
            # Choose two random parents
            p1 = np.random.randint(0, self.elitePopulations)
            p2 = np.random.randint(0, self.elitePopulations)

            while p1 == p2:
                p2 = np.random.randint(0, self.elitePopulations)
    
            parent1 = elite[p1]
            parent2 = elite[p2]
            child = []

            for i in range(self.utilitySize):
                if(i % 2 == 0):
                    child.append(parent2[i])
                else:
                    child.append(parent1[i])
            #print(child)
            self.utilityFunction.append(child)

    def mutation(self):
        print("mutation")

        for x in range(self.populationSize):
            mutate = np.random.randint(0, self.utilitySize)
            mutationValue = np.random.normal(0, self.mutationSTD)
            self.utilityFunction[x][mutate] += mutationValue

    def calcUtilityValue(self, i, states):
        util = self.utilityFunction[i]
        value = 0
        for x in states:
            if x == 2 or x == 4 or x == 10: # negative
                value -= util[x]
            else:
                value += util[x]
        return value

    def chooseMove(self, i, move_pieces, player_pieces, dice, enemy_pieces):
        #print("dice:",dice)
        
        enemy_pieces = self.adjustEnemyPieces(enemy_pieces)
        #print("enemy_pieces", enemy_pieces)
        utilityValues = []
        #print(move_pieces)
        for x in move_pieces:
            brickPosition = player_pieces[x]
            #print("brickPosition", brickPosition)
            states = self.determineStateSpace(brickPosition,dice,enemy_pieces)
            #print(states)
            ## Calc utility value
            util = self.calcUtilityValue(i,states)
            utilityValues.append(util)

        # Choose highste utility value and return from move pieces
        maxValue = max(utilityValues)
        index = utilityValues.index(maxValue)
        piece_to_move = move_pieces[index]
        return piece_to_move

    def playGame(self,i):
            
        import ludopy

        win = 0

        g = ludopy.Game()
        there_is_a_winner = False

        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
            there_is_a_winner), player_i = g.get_observation()
            
            if len(move_pieces):
                if player_i == 0:
                    #print("Round:",g.round)
                    piece_to_move = self.chooseMove(i,move_pieces,player_pieces,dice, enemy_pieces)
                    win = 1
                    #piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                else:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                    win = 0
            else:
                piece_to_move = -1

            _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)


        #print("Saving history to numpy file")
        #g.save_hist("game_history.npy")
        #print("Saving game video")
        #g.save_hist_video("game_video.mp4")

        return win

    def playGameSaveVideo(self, i):

        import ludopy

        win = 0

        g = ludopy.Game()
        there_is_a_winner = False

        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
             there_is_a_winner), player_i = g.get_observation()

            if len(move_pieces):
                if player_i == 0:
                    # print("Round:",g.round)
                    piece_to_move = self.chooseMove(i, move_pieces, player_pieces, dice, enemy_pieces)
                    win = 1
                    # piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                else:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                    win = 0
            else:
                piece_to_move = -1

            _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

        print("Saving history to numpy file")
        g.save_hist("game_history.npy")
        print("Saving game video")
        g.save_hist_video("game_video.mp4")

        return win

    def computeFitness(self):
        score = []
        # Test the population
        for i in range(self.populationSize):
            if (i % 10 == 0):
                print("Individual",i)
            #Play 100 game
            winCounter = 0
            for x in range(self.trainingGames):
                winCounter += self.playGame(i)
            #print(i, winCounter)
            score.append(winCounter)
        return score
        
    def succesReached(self, score):
        succesCriteria = self.trainingGames * self.succesCriteria
        maxValue = max(score)
        print("Best score", maxValue)
        meanValue = np.mean(score)
        print("Mean win rate", meanValue)
        if self.bestScoreOverall < maxValue:
            self.bestScoreOverall = maxValue
            self.bestInduvidual = score.index(maxValue)
            self.bestUtil = self.utilityFunction[self.bestInduvidual]
        if succesCriteria <= maxValue:
            return True
        return False
    
def openCVS():
    with open ('data.csv', 'w'):
        pass

def trainGA():

    df = pd.DataFrame()

    ga = geneticAlgorithm(100)

    #enemy_pieces = [[11, 7, 0, 1],
    #                [ 0,  0,  0, 25],
    #                [24,  0,  0,  0]]

    #ga.determineStateSpace(1, 5, enemy_pieces)

    generations = 0

    ga.initPopulation()

    score = ga.computeFitness()

    while not ga.succesReached(score) and generations <= 25:
        df[str(generations)] = score
        df.to_csv("data.csv")
        #print(df.head)
        elite = ga.selection(score)
        ga.crossover(elite)
        ga.mutation()
        score = ga.computeFitness()

        generations += 1

    print("generations",generations)
    df.to_csv("data.csv")
    print("Best score", ga.bestScoreOverall)
    meanValue = np.mean(score)
    print("Mean win rate", meanValue)
    print(ga.bestUtil)

    # Play a game with the best
    win = ga.playGameSaveVideo(ga.bestInduvidual)
    while not win:
        win = ga.playGameSaveVideo(ga.bestInduvidual)

    return True

if __name__ == '__main__':
    #unittest.main()
    openCVS()
    trainGA()
