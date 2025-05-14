def get_user_input(test_mode=True):
    """
    Get user input for the model.

    Parameters:
        test_mode (bool): If True, use default test values.

    Returns:
        dict: User inputs including target date, species, generations, and output formats.
    """
    if test_mode:
        return {
            {
                "user_email": "jon@test.com",
                "detection_date": "2023-03-15",
                "species": "mexfly",
                "generations": 3,
                "output_formats": ["json"],
            },
            {
                "user_email": "jon@test.com",
                "detection_date": "2025-02-15",
                "species": "off",
                "generations": 3,
                "output_formats": ["json"],
            },
        }

    # In a real application, you would replace this with actual user input logic
    return {
        "target_date": input("Enter target date (YYYY-MM-DD): "),
        "species": input("Enter species: "),
        "generations": int(input("Enter number of generations: ")),
        "output_formats": input("Enter output formats (comma-separated): ").split(","),
    }
