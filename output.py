from datetime import timedelta
import json


class OutputData:
    # ...existing code...

    def _extract_point(self, coordinate):
        """
        Helper method to extract data from the coordinate.
        Assumes `coordinate` is a neighborhood window and extracts the central point.
        """
        # Example logic to extract the central point from a neighborhood window
        return coordinate[len(coordinate) // 2]

    def to_json(self, start_date, species, coordinates):
        """
        Generate a JSON representation of the output data.
        Includes start date, species, coordinates, and F1, F2, F3 generation dates.
        """
        central_point = self._extract_point(coordinates)
        f1_date = start_date + timedelta(days=self.f1_days)
        f2_date = start_date + timedelta(days=self.f2_days)
        f3_date = start_date + timedelta(days=self.f3_days)

        output_json = {
            "start_date": start_date.isoformat(),
            "species": species,
            "coordinates": central_point,
            "generations": {
                "F1": f1_date.isoformat(),
                "F2": f2_date.isoformat(),
                "F3": f3_date.isoformat(),
            },
        }
        return json.dumps(output_json, indent=4)
