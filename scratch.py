import time

# Function to calculate EMA
def calculate_ema(current_value, previous_ema, alpha=0.9):
    return alpha * current_value + (1 - alpha) * previous_ema

# Generate an exponential list of 20 numbers
numbers = [2**i for i in range(20)]
# numbers = [2**(10 - i) for i in range(10)] + [3**1] * 30
# numbers = [5 for _ in range(20)]

# Initialize EMA with the first number
ema = numbers[0]

# Loop through the list and perform calculations
for i, number in enumerate(numbers):
    # Print the number
    print(f"Number: {number}")
    
    # Calculate EMA
    if i > 0:  # skip EMA update for the first number
        ema = calculate_ema(number, ema)
    
    print(f"EMA: {ema}")
    
    # Calculate number divided by EMA
    div_by_ema = number / ema
    print(f"Number / EMA: {div_by_ema}")
    
    # Calculate number minus EMA
    minus_ema = number - ema*0.9
    print(f"Number - EMA: {minus_ema}")
    
    # Pause for 1 second
    time.sleep(1)
