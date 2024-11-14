## PROJECT-PHASE-5
# WAKULIMA FARMTECH
A simple ML and DL based website which recommends the best crop to grow, fertilizers to use and the diseases caught by your crops.

# MOTIVATION
* Farming is one of the major sectors that influences a country’s economic     
   growth.
* In countries in Africa, majority of the population is dependent on   
  agriculture for their livelihood. Many new technologies, such as Machine 
  Learning and Deep Learning, are being implemented into agriculture so that 
  it is easier for farmers to grow and maximize their yield.
  
* In this project, we present a website in which the following applications are implemented; Crop recommendation, Fertilizer recommendation and Plant disease prediction, respectively.

  * In the crop recommendation application, the user can provide the soil data 
   from their side and the application will predict which crop should the user 
   grow.

  * For the fertilizer recommendation application, the user can input the soil 
   data and the type of crop they are growing, and the application will 
   predict what the soil lacks or has excess of and will recommend 
   improvements.

  * For the last application, that is the plant disease prediction 
    application,still in development.

 ## Among the challenges farmers face include:

1. Crop Selection Uncertainty: Farmers may not know which crops are best suited for their soil and climate conditions, leading to poor yields and economic loss.
2. Fertilizer Mismanagement: Incorrect use of fertilizers can result in soil nutrient imbalances, affecting crop health and yield. Farmers may not know the right type or amount of fertilizer required for their crops.
3. Plant Disease Identification: Identifying plant diseases can be difficult for farmers, especially without expert knowledge. Delayed or incorrect diagnosis can lead to severe crop damage and lower productivity.

To enhance decision-making, a data-driven recommendation system is needed that provides personalized, actionable insights.

## Objective
The primary goal of this project is to develop an intelligent crop and fertilizer recommendation system that assists farmers in optimizing their farming practices. The system will recommend the most suitable crops and fertilizers based on factors such as soil composition, crop type and weather patterns. By leveraging machine learning and data analytics, the system aims to improve crop yields, reduce costs, and promote sustainable agricultural practices.

## How to use 💻
- Crop Recommendation system ==> enter the corresponding nutrient values of your soil, state and city. Note that, the N-P-K (Nitrogen-Phosphorous-Pottasium) values to be entered should be the ratio between them.
Note: When you enter the city name, make sure to enter mostly common city names. Remote cities/towns may not be available in the [Weather API](https://openweathermap.org/) from where humidity, temperature data is fetched.

- Fertilizer suggestion system ==> Enter the nutrient contents of your soil and the crop you want to grow. The algorithm will tell which nutrient the soil has excess of or lacks. Accordingly, it will give suggestions for buying fertilizers.

- Disease Detection System ==> Upload/capture an image of leaf of your plant. The algorithm will tell the crop type and whether it is diseased or healthy. If it is diseased, it will tell you the cause of the disease and suggest you how to prevent/cure the disease accordingly.
Note that, for now it only supports following crops

<details>
  <summary>Supported crops
</summary>

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Pepper
- Orange
- Peach
- Potato
- Soybean
- Strawberry
- Tomato
- Squash
- Raspberry
</details>

## How to run locally 🛠️
- Before the following steps make sure you have [git](https://git-scm.com/download), [Anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system
- Clone the complete project with `git clone https://github.com/samuelMG1/PROJECT-PHASE-5.git 
## Demo
[Watch the demo video](https://drive.google.com/drive/folders/1ie7epl-nmU5GrWTKdrHAbsW82asxKjJ9?usp=sharing)

  
## Solutions
The solutions the recommendation sytem will provide include:

1. Crop Recommendation: By using machine learning to analyze soil data provided by the user, your application can predict the most suitable crops for a specific soil type, enabling farmers to make informed decisions and improve yields.
2. Fertilizer Recommendation: Based on the user's soil data and the type of crop they are growing, the application can recommend the appropriate fertilizer by identifying any deficiencies or excess nutrients in the soil, ensuring better crop growth and healthier soil.
3. Plant Disease Prediction: The image recognition feature allows users to upload images of diseased plant leaves. The application then predicts the disease and offers background information and treatment suggestions, enabling timely and effective intervention.

## Benefits of the recommendation system

The crop-fertilizer recommendation system will provide solutions to some of the major challenges faced by farmers. The benefits of using our recommendation system include:

- Increased Crop Yield: Farmers will receive precise recommendations for crops and fertilizers, leading to significant improvements in crop productivity.
- Cost Efficiency: By optimizing fertilizer use and avoiding over-application, farmers can reduce costs while maintaining or increasing yields.
- Sustainability: The system will promote responsible fertilizer use, reducing the risk of soil degradation and environmental pollution.
- Personalized Recommendations: Tailored insights based on specific farm conditions (soil properties, climate, and crop type) ensure relevant and actionable advice for each farmer.
- Disease Management: By integrating plant disease prediction features, the system can help farmers identify and treat crop diseases in a timely manner, minimizing losses.
