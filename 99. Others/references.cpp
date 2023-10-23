#include <iostream>
#include <string>

int main() {
    std::string food = "Pizza"; // food variable
    std::string &meal = food; // reference to food

    food = "Potato"; // food variable
    
    std::cout << meal << "\n";
    std::cout << &meal << "\n";

    return 0;
}