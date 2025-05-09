# main.py
# Adjusted imports to reflect the new location of main.py in src
from src.io_handlers.config import load_config
from src.core.weather import WeatherDataHandler
from src.io_handlers.user_input import get_user_input
from src.utils.species_params import load_species_params
from src.core.model import DegreeDayModel
from src.io_handlers.output_generator import OutputGenerator

def main():
   
    # Load configuration
    config = load_config("../config/settings.yaml")
    inputs = get_user_input(test_mode=True)  # CLI/GUI/web form

    for input in inputs:
        # Check if the input is valid
  
        # Extract parameters
        target_date = input.get('detection_date')
        species = input.get('species')
        generations = input.get('generations')
        output_formats = input.get('output_formats', [])
        
        # Validate required parameters # replace with actual validation logic 
        if not target_date or not species or not generations:
            raise ValueError("Missing required parameters: detection_date, species, generations.")
        
        species_params = load_species_params(
            species=species,
            data_path="../config/fly_models.json"
        )
        # ----------------------------
        #weather loading
        # ----------------------------
        weather =  
    


    # Initialize core components
    weather= WeatherDataHandler(config['weather'])
    # Load species parameters
    
    model = DegreeDayModel(species_params)
    output = OutputGenerator(config['output'])
    
   
    # ----------------------------
    # 3. DATA PREPARATION
    # ----------------------------
    # Get weather data
    current_data = weather.get_recent_data(
        start_date=inputs['target_date'],
        days_back=30  # Example: last 30 days
    )
    
    historical_data = weather.get_historical_data(
        years=range(2000, 2023)  # Example range
    )
    
    # ----------------------------
    # 4. MODEL EXECUTION
    # ----------------------------
    results = model.run(
        current_data=current_data,
        historical_data=historical_data,
        detection_date=pd.Timestamp(inputs['target_date']),
        species=inputs['species'],
        generations=inputs['generations']
    )
    
    # ----------------------------
    # 5. OUTPUT GENERATION
    # ----------------------------
    if "plot" in inputs['output_formats']:
        output.generate_plots(
            results,
            filename=f"{inputs['species']}_generation_{inputs['target_date']}.png"
        )
    
    if "json" in inputs['output_formats']:
        output.save_json(
            results,
            filename=f"{inputs['species']}_results.json"
        )
    
    if "report" in inputs['output_formats']:
        output.generate_report(
            results,
            species=inputs['species'],
            date=inputs['target_date']
        )
    
    print("Pipeline executed successfully!")


if __name__ == "__main__":
    main()