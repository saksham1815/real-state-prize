import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    if __model is None:
        print("Model is not loaded. Please load the model first.")
        return None
    
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    try:
        return round(__model.predict([x])[0], 2)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    try:
        with open(r"C:\Users\saxen\Downloads\Telegram Desktop\real estate price prediction\server\artifects\columns.json", "r") as f:
            __data_columns = json.load(f)['data_columns']
            __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk
    except FileNotFoundError as e:
        print(f"Error loading columns.json: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding columns.json: {e}")
        return

    global __model
    if __model is None:
        try:
            with open(r'C:\Users\saxen\Downloads\Telegram Desktop\real estate price prediction\server\artifects\bangalore_home_prices_model.pickle', 'rb') as f:
                __model = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            return
        except pickle.UnpicklingError as e:
            print(f"Error unpickling model: {e}")
            return

    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print("Available locations:", get_location_names())
    print("Estimated price for 1st Phase JP Nagar (1000 sqft, 3 BHK, 3 bath):", get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print("Estimated price for 1st Phase JP Nagar (1000 sqft, 2 BHK, 2 bath):", get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print("Estimated price for Kalhalli (1000 sqft, 2 BHK, 2 bath):", get_estimated_price('Kalhalli', 1000, 2, 2))  # other location
    print("Estimated price for Ejipura (1000 sqft, 2 BHK, 2 bath):", get_estimated_price('Ejipura', 1000, 2, 2))  # other location
