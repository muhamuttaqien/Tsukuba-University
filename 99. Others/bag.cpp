#include <iostream>
#include <vector>

class Bag {
    private: 
        std::vector<int> items;

    public:
        void addItem(int item) {
            items.push_back(item);
        }

        void displayItems() {
            for (const auto& item: items) {
                std::cout << item << " ";
            }
            std::cout << std::endl;
        }
};

int main() {
    Bag myBag;

    myBag.addItem(1);
    myBag.addItem(2);
    myBag.addItem(3);

    std::cout << "Items in the Bag: ";
    myBag.displayItems();

    return 0;
}