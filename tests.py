import unittest
import json
from app import app

class FlaskTestCase(unittest.TestCase):

    # Test for the predict route with valid input
    def test_predict(self):
        tester = app.test_client(self)
        response = tester.post(
            '/predict',
            data=json.dumps({
                'area': 2000,
                'bedrooms': 3,
                'bathrooms': 2,
                'stories': 1,
                'mainroad': 'yes',
                'guestroom': 'no',
                'basement': 'yes',
                'hotwaterheating': 'no',
                'airconditioning': 'yes',
                'parking': 1,
                'prefarea': 'yes',
                'furnishingstatus': 'furnished'
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('predicted_price', json.loads(response.data))

    # Test for the predict route with invalid input
    def test_invalid_input(self):
        tester = app.test_client(self)
        response = tester.post(
            '/predict',
            data=json.dumps({
                'area': 'not_a_number',  # Invalid input
                'bedrooms': 3,
                'bathrooms': 2,
                'stories': 1,
                'mainroad': 'yes',
                'guestroom': 'no',
                'basement': 'yes',
                'hotwaterheating': 'no',
                'airconditioning': 'yes',
                'parking': 1,
                'prefarea': 'yes',
                'furnishingstatus': 'furnished'
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)  # Expecting a 400 for bad input

if __name__ == '__main__':
    unittest.main()
