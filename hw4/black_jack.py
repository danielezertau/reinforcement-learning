import os
import random
import numpy as np

deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]*4
GAMMA = 1
FINAL_EPS = 0.01
NUM_STATES = 32
NUM_ACTIONS = 2
v_hat = np.zeros(NUM_STATES + 1)
num_visits = np.zeros((NUM_STATES + 1, NUM_ACTIONS))

q_hat = np.zeros((NUM_STATES + 1, NUM_ACTIONS))


def play_dealer(dealer_hand):
    while total(dealer_hand) <= 15:
        hit(dealer_hand)


class Schedule:
    def __init__(self, initial_eps=1, final_eps=0.01, anneal_every=1000):
        self.anneal_every = anneal_every
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.current_eps = initial_eps

    def get_epsilon(self, t):
        if t % self.anneal_every == 0:
            self.current_eps = max(1 / (np.power(t, (2/3)) / self.anneal_every), self.final_eps)
        return self.current_eps


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
    deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]*4


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


def print_results(dealer_hand, player_hand, debug=False):
    if debug:
        print("The dealer has a " + str(dealer_hand) + " for a total of " + str(total(dealer_hand)))
        print("You have a " + str(player_hand) + " for a total of " + str(total(player_hand)))


def blackjack(dealer_hand, player_hand, debug=False):
    if total(player_hand) == 21:
        print_results(dealer_hand, player_hand)
        if debug:
            print("Congratulations! You got a Blackjack!\n")
        return True
    elif total(dealer_hand) == 21:
        print_results(dealer_hand, player_hand)
        if debug:
            print("Sorry, you lose. The dealer got a blackjack.\n")
        return True
    else:
        return False


def double_aces(dealer_hand, player_hand, debug=False):
    if total(player_hand) == 22:
        print_results(dealer_hand, player_hand)
        if debug:
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


def score(dealer_hand, player_hand, debug=False):
    if total(player_hand) == 21:
        print_results(dealer_hand, player_hand)
        if debug:
            print("Congratulations! You got a Blackjack!\n")
        game_result = "win"
    elif total(dealer_hand) == 21:
        print_results(dealer_hand, player_hand)
        if debug:
            print("Sorry, you lose. The dealer got a blackjack.\n")
        game_result = "lose"
    elif total(player_hand) > 21:
        print_results(dealer_hand, player_hand)
        if debug:
            print("Sorry. You busted. You lose.\n")
        game_result = "lose"
    elif total(dealer_hand) > 21:
        print_results(dealer_hand, player_hand)
        if debug:
            print("Dealer busts. You win!\n")
        game_result = "win"
    elif total(player_hand) < total(dealer_hand):
        print_results(dealer_hand, player_hand)
        if debug:
            print("Sorry. Your score isn't higher than the dealer. You lose.\n")
        game_result = "lose"
    elif total(player_hand) > total(dealer_hand):
        print_results(dealer_hand, player_hand)
        if debug:
            print("Congratulations. Your score is higher than the dealer. You win\n")
        game_result = "win"
    else:
        print_results(dealer_hand, player_hand)
        if debug:
            print("It's a tie!\n")
        game_result = "tie"
    return get_reward(game_result)


def game_td0(num_games):
    for _ in range(num_games):
        player_hand, dealer_hand = game_start_and_deal()
        if check_bust_on_start(player_hand, dealer_hand):
            continue

        # Gambler
        while total(player_hand) < 18:
            state = total(player_hand)
            action = 1
            reward = 0
            hit(player_hand)
            next_state = total(player_hand)
            td0(state, action, reward, next_state, next_state > 21)
        if total(player_hand) > 21:
            reset_deck()
            continue
        # Dealer
        play_dealer(dealer_hand)

        state = total(player_hand)
        action = 0
        # Check who won
        reward = score(dealer_hand, player_hand)
        next_state = state
        # Perform TD step
        td0(state, action, reward, next_state, True)
        reset_deck()


def game_start_and_deal(debug=False):
    if debug:
        clear()
        print("WELCOME TO BLACKJACK!\n")
    dealer_hand = deal()
    player_hand = deal()
    if debug:
        print("The dealer is showing a " + str(dealer_hand[0]))
        print("You have a " + str(player_hand) + " for a total of " + str(total(player_hand)))
    return player_hand, dealer_hand


def check_bust_on_start(player_hand, dealer_hand):
    if blackjack(dealer_hand, player_hand) or double_aces(dealer_hand, player_hand):
        reset_deck()
        return True
    return False


def game_sarsa(num_games, schedule):
    for t in range(1, num_games + 1):
        player_hand, dealer_hand = game_start_and_deal()
        if check_bust_on_start(player_hand, dealer_hand):
            continue
        # Gambler
        done = bust = False
        while not done and not bust:
            state = total(player_hand)
            action = pi(state, t, schedule)
            # Hit
            if action == 1:
                reward = 0
                hit(player_hand)
                next_state = total(player_hand)
                sarsa(state, action, reward, next_state, t, bust, schedule)
                if next_state > 21:
                    bust = True
            else:
                done = True

        # Dealer
        play_dealer(dealer_hand)

        state = total(player_hand)
        # Check who won
        reward = score(dealer_hand, player_hand)
        if not bust:
            next_state = state
            sarsa(state, action, reward, next_state, t, True, schedule)

        reset_deck()


def td0(state, action, reward, next_state, done):
    num_visits[state][action] += 1
    alpha = 1 / num_visits[state][action]
    v_next = 0 if done else v_hat[next_state]
    v_hat[state] += alpha * (reward + GAMMA * v_next - v_hat[state])


def sarsa(state, action, reward, next_state, t, bust, sched):
    next_action = pi(next_state, t, sched)
    num_visits[state][action] += 1
    alpha = 1 / num_visits[state][action]
    q_next = 0 if bust else q_hat[next_state][next_action]
    q_hat[state, action] += alpha * (reward + GAMMA * q_next - q_hat[state][action])


def pi(state, t, schedule):
    eps = schedule.get_epsilon(t)
    if random.uniform(0, 1) <= eps:
        return np.random.randint(NUM_ACTIONS)
    else:
        return np.argmax(q_hat[state])


def q1(num_games):
    game_td0(num_games)
    # Fix zero division error
    num_visits[num_visits == 0] = 1
    states_visits_frac = np.sum(num_visits / num_games, axis=1)
    win_probs = (v_hat * states_visits_frac)
    print("TD0")
    print(f"The probability of winning is {np.sum(win_probs)}")


def q2(num_games):
    eps_schedule = Schedule()
    game_sarsa(num_games, eps_schedule)
    # Fix zero division error
    num_visits[num_visits == 0] = 1
    action_choice_frac = (num_visits.T / np.sum(num_visits, axis=1)).T
    v_from_q = np.sum(q_hat * action_choice_frac, axis=1)
    states_visits_frac = np.sum(num_visits / num_games, axis=1)
    win_probs = (v_from_q * states_visits_frac)
    print("-" * 100)
    print("SARSA")
    for st, prob in enumerate(v_from_q[4:22]):
        opt_action = "hit" if np.argmax(q_hat[st + 4]) == 1 else "stay"
        print(f"State {st + 4}: winning prob: {prob:.5f}, optimal action: {opt_action}")
    print(f"Total win prob: {np.sum(win_probs)}")


if __name__ == "__main__":
    n = 10000000
    q1(n)
    q2(n)
