import os
import random
import numpy as np

deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]*4
GAMMA = 1
FINAL_EPS = 0.01
NUM_STATES = 31
NUM_ACTIONS = 2
v_hat = np.zeros(NUM_STATES + 1)
num_visits = np.zeros((NUM_STATES + 1, NUM_ACTIONS))

q_hat = np.zeros((NUM_STATES + 1, NUM_ACTIONS))


def deal():
    hand = []
    for i in range(2):
        random.shuffle(deck)
        card = deck.pop()
        if card == 10:
            card = "FACE"
        if card == 11:
            card = "A"
        hand.append(card)
    return hand


def reset_deck():
    global deck
    deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]*4


def total(hand):
    hand_value = 0
    for card in hand:
        if card == "FACE":
            hand_value += 10
        elif card == "A":
            hand_value += 11
        else:
            hand_value += card
    return hand_value


def hit(hand):
    card = deck.pop()
    if card == 10:
        card = "FACE"
    if card == 11:
        card = "A"
    hand.append(card)
    return hand


def clear():
    if os.name == 'nt':
        os.system('CLS')
    if os.name == 'posix':
        os.system('clear')


def print_results(dealer_hand, player_hand):
    print("The dealer has a " + str(dealer_hand) + " for a total of " + str(total(dealer_hand)))
    print("You have a " + str(player_hand) + " for a total of " + str(total(player_hand)))


def blackjack(dealer_hand, player_hand):
    if total(player_hand) == 21:
        print_results(dealer_hand, player_hand)
        print("Congratulations! You got a Blackjack!\n")
        return True
    elif total(dealer_hand) == 21:
        print_results(dealer_hand, player_hand)
        print("Sorry, you lose. The dealer got a blackjack.\n")
        return True
    else:
        return False


def double_aces(dealer_hand, player_hand):
    if total(player_hand) == 22:
        print_results(dealer_hand, player_hand)
        print("Sorry, you lose. You got double aces\n")
        return True
    else:
        return False


def get_reward(game_result):
    if game_result == "win":
        return 1
    elif game_result == "tie":
        return 0
    elif game_result == "lose":
        return 0


def score(dealer_hand, player_hand):
    if total(player_hand) == 21:
        print_results(dealer_hand, player_hand)
        print("Congratulations! You got a Blackjack!\n")
        game_result = "win"
    elif total(dealer_hand) == 21:
        print_results(dealer_hand, player_hand)
        print("Sorry, you lose. The dealer got a blackjack.\n")
        game_result = "lose"
    elif total(player_hand) > 21:
        print_results(dealer_hand, player_hand)
        print("Sorry. You busted. You lose.\n")
        game_result = "lose"
    elif total(dealer_hand) > 21:
        print_results(dealer_hand, player_hand)
        print("Dealer busts. You win!\n")
        game_result = "win"
    elif total(player_hand) < total(dealer_hand):
        print_results(dealer_hand, player_hand)
        print("Sorry. Your score isn't higher than the dealer. You lose.\n")
        game_result = "lose"
    elif total(player_hand) > total(dealer_hand):
        print_results(dealer_hand, player_hand)
        print("Congratulations. Your score is higher than the dealer. You win\n")
        game_result = "win"
    else:
        print_results(dealer_hand, player_hand)
        print("It's a tie!\n")
        game_result = "tie"
    return get_reward(game_result)


def game(num_games):
    for t in range(1, num_games + 1):
        clear()
        print("WELCOME TO BLACKJACK!\n")
        dealer_hand = deal()
        player_hand = deal()
        print("The dealer is showing a " + str(dealer_hand[0]))
        print("You have a " + str(player_hand) + " for a total of " + str(total(player_hand)))
        if blackjack(dealer_hand, player_hand) or double_aces(dealer_hand, player_hand):
            reset_deck()
            continue
        # Gambler
        done = bust = False
        while not done and not bust:
            state = total(player_hand)
            action = pi(state, t)
            # Hit
            if action == 1:
                reward = 0
                hit(player_hand)
                next_state = total(player_hand)
                sarsa(state, action, reward, next_state, t)
                if next_state > 21:
                    bust = True
            else:
                done = True

        # Dealer
        while total(dealer_hand) <= 15 and not bust:
            hit(dealer_hand)

        state = total(player_hand)
        # Check who won
        reward = score(dealer_hand, player_hand)
        if not bust:
            next_state = state
            sarsa(state, action, reward, next_state, t)

        reset_deck()


def td0(state, action, reward, next_state):
    num_visits[state][action] += 1
    alpha = 1 / num_visits[state][action]
    v_hat[state] += alpha * (reward + GAMMA * v_hat[next_state] - v_hat[state])


def sarsa(state, action, reward, next_state, t):
    next_action = pi(next_state, t)
    num_visits[state][action] += 1
    alpha = 1 / num_visits[state][action]
    q_hat[state, action] += alpha * (reward + GAMMA * q_hat[next_state][next_action] - q_hat[state][action])


def get_epsilon(t):
    return max(1 / t, FINAL_EPS)


def pi(state, t):
    eps = get_epsilon(t)
    if random.uniform(0, 1) <= eps:
        return np.random.randint(NUM_ACTIONS)
    else:
        return np.argmax(q_hat[state])


if __name__ == "__main__":
    n = 10000
    game(n)
    # Fix zero division error
    num_visits[num_visits == 0] = 1
    print(q_hat[4:22, :])
    print(np.argmax(q_hat[4:22, :], axis=1))
    # print(v_hat * (1 / np.sum(num_visits, axis=1)))
