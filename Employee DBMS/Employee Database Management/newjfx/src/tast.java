import javafx.application.Application;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.*;
import javafx.stage.Stage;
import java.io.File;
import java.sql.*;
import javafx.scene.control.ComboBox;

public class tast extends Application {
    private Button insertButton, viewButton, updateButton, deleteButton, searchByNameButton, printDataButton; // New button added

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Employee Database");

        // Initialize components
        insertButton = new Button("Insert New Data");
        viewButton = new Button("View Employee Details");
        updateButton = new Button("Update Data");
        deleteButton = new Button("Delete Data");
        searchByNameButton = new Button("Search by name");
        printDataButton = new Button("Print All Data"); // New button added

        // Set button styles
        insertButton.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white;");
        viewButton.setStyle("-fx-background-color: #3498db; -fx-text-fill: white;");
        updateButton.setStyle("-fx-background-color: #f39c12; -fx-text-fill: white;");
        deleteButton.setStyle("-fx-background-color: #e74c3c; -fx-text-fill: white;");
        searchByNameButton.setStyle("-fx-background-color: #9b59b6; -fx-text-fill: white;");
        printDataButton.setStyle("-fx-background-color: #34495e; -fx-text-fill: white;"); // Styling for new button

        // Set layout
        VBox layout = new VBox(10);
        layout.setStyle("-fx-background-color: #ecf0f1; -fx-padding: 10px;");
        layout.getChildren().addAll(insertButton, viewButton, updateButton, deleteButton, searchByNameButton, printDataButton); // Added new button

        // Add action listeners
        insertButton.setOnAction(e -> {
            primaryStage.close();
            new InsertFrame().start(new Stage());
        });

        viewButton.setOnAction(e -> {
            primaryStage.close();
            new ViewFrame().start(new Stage());
        });

        updateButton.setOnAction(e -> {
            // Open a new window for updating data
            new UpdateFrame().start(new Stage());
        });

        deleteButton.setOnAction(e -> {
            // Open a dialog to get ID for deleting data
            TextInputDialog dialog = new TextInputDialog();
            dialog.setTitle("Delete Data");
            dialog.setHeaderText("Enter ID to delete:");
            dialog.setContentText("ID:");
            dialog.showAndWait().ifPresent(id -> {
                int employeeID = Integer.parseInt(id);
                deleteData(employeeID);
            });
        });

        searchByNameButton.setOnAction(e -> {
            // Open a dialog to get name for searching data
            TextInputDialog dialog = new TextInputDialog();
            dialog.setTitle("Search by Name");
            dialog.setHeaderText("Enter name to search:");
            dialog.setContentText("Name:");
            dialog.showAndWait().ifPresent(name -> {
                searchByName(name);
            });
        });

        printDataButton.setOnAction(e -> printAllData()); // Action listener for the new button

        // Set scene
        primaryStage.setScene(new Scene(layout, 400, 230));
        primaryStage.show();

        // Check if the database file exists, if not, create it
        String dbFilePath = "C:/Users/syedr/OneDrive/Desktop/rayan/Projects/Employee Database Management/Database21.accdb";
        File dbFile = new File(dbFilePath);

        if (!dbFile.exists()) {
            createDatabase(dbFilePath);
        }

    }
    

    // Method to create the database file
    private void createDatabase(String filePath) {
        try {
            String databaseURL = "jdbc:ucanaccess://" + filePath + ";newDatabaseVersion=V2010";
            try (Connection conn = DriverManager.getConnection(databaseURL)) {
                // Create table
                String createTableSQL = "CREATE TABLE Employees (" +
                        "id INT PRIMARY KEY," +
                        "first_name VARCHAR(255)," +
                        "middle_name VARCHAR(255)," +
                        "last_name VARCHAR(255)," +
                        "position VARCHAR(255)," +
                        "salary VARCHAR(255)," +   // fixed
                        "age DOUBLE," +
                        "gender VARCHAR(10)," +
                        "phno VARCHAR(225)," +
                        "addr VARCHAR(255)," +
                        "Email VARCHAR(255))";
                try (Statement stmt = conn.createStatement()) {
                    stmt.executeUpdate(createTableSQL);
                }
            }
            showAlert("Database file created successfully.");
        } catch (SQLException ex) {
            showAlert("Error creating database file: " + ex.getMessage());
        }
    }


    // Method to delete employee data
    private void deleteData(int id) {
        try (Connection conn = DriverManager.getConnection("jdbc:ucanaccess://C:/Users/syedr/OneDrive/Desktop/rayan/Projects/Employee Database Management/Database21.accdb");
             PreparedStatement pstmt = conn.prepareStatement("DELETE FROM Employees WHERE id=?")) {

            pstmt.setInt(1, id);

            int rowsAffected = pstmt.executeUpdate();
            showAlert(rowsAffected + " row(s) deleted successfully.");
        } catch (SQLException ex) {
            showAlert("Error deleting data: " + ex.getMessage());
        }
    }

    // Method to search employees by name
    // Method to search employees by name
    // Method to search employees by name
    private void searchByName(String fullName) {
        // Splitting the input into first name, middle name, and last name
        String[] names = fullName.split("\\s+");
        String firstName = "";
        String middleName = "";
        String lastName = "";

        if (names.length >= 1) {
            firstName = names[0];
        }
        if (names.length >= 2) {
            middleName = names[1];
        }
        if (names.length >= 3) {
            lastName = names[2];
        }

        try (Connection conn = DriverManager.getConnection("jdbc:ucanaccess://C:/Users/syedr/OneDrive/Desktop/rayan/Projects/Employee Database Management/Database21.accdb");
             PreparedStatement pstmt = conn.prepareStatement("SELECT * FROM Employees WHERE (first_name LIKE ? OR middle_name LIKE ? OR last_name LIKE ?) AND (first_name LIKE ? OR middle_name LIKE ? OR last_name LIKE ?) AND (first_name LIKE ? OR middle_name LIKE ? OR last_name LIKE ?)")) {

            pstmt.setString(1, "%" + firstName + "%");
            pstmt.setString(2, "%" + firstName + "%");
            pstmt.setString(3, "%" + firstName + "%");
            pstmt.setString(4, "%" + middleName + "%");
            pstmt.setString(5, "%" + middleName + "%");
            pstmt.setString(6, "%" + middleName + "%");
            pstmt.setString(7, "%" + lastName + "%");
            pstmt.setString(8, "%" + lastName + "%");
            pstmt.setString(9, "%" + lastName + "%");

            ResultSet rs = pstmt.executeQuery();

            StringBuilder result = new StringBuilder();
            boolean found = false;

            while (rs.next()) {
                found = true;
                int id = rs.getInt("id");
                String foundFirstName = rs.getString("first_name");
                String foundMiddleName = rs.getString("middle_name");
                String foundLastName = rs.getString("last_name");
                String position = rs.getString("position");
                String salary = rs.getString("salary");
                double age = rs.getDouble("age");
                String gender = rs.getString("gender");
                String phno = rs.getString("phno");
                String addr = rs.getString("addr");
                String email = rs.getString("Email"); // Changed postal_code to Email

                result.append("ID: ").append(id).append("\n")
                        .append("First Name: ").append(foundFirstName).append("\n")
                        .append("Middle Name: ").append(foundMiddleName).append("\n")
                        .append("Last Name: ").append(foundLastName).append("\n")
                        .append("Position: ").append(position).append("\n")
                        .append("Salary: ").append(salary).append("\n")
                        .append("Age: ").append(age).append("\n")
                        .append("Gender: ").append(gender).append("\n")
                        .append("Phone Number: ").append(phno).append("\n")
                        .append("Address: ").append(addr).append("\n")
                        .append("Email: ").append(email).append("\n\n"); // Changed postal_code to Email
            }

            if (found) {
                showAlert("Search Results:\n\n" + result.toString());
            } else {
                showAlert("No employees found with the provided name combination.");
            }
        } catch (SQLException ex) {
            showAlert("Error searching data: " + ex.getMessage());
        }
    }


    class InsertFrame extends Application {
        private TextField idField, firstNameField, middleNameField, lastNameField, positionField, salaryField, phnoField, addrField, postalCodeField;
        private ComboBox<String> currencyComboBox; // Add ComboBox for currency selection
        private ComboBox<Integer> ageComboBox; // Adding ComboBox for age selection
        private ComboBox<String> genderComboBox;
        private ComboBox<String> countryCodeComboBox; // ComboBox for country codes
        private Button insertButton, searchButton;

        @Override
        public void start(Stage primaryStage) {
            primaryStage.setTitle("Insert Data");

            // Initialize components
            idField = new TextField();
            firstNameField = new TextField();
            middleNameField = new TextField();
            lastNameField = new TextField();
            positionField = new TextField();
            salaryField = new TextField();
            phnoField = new TextField();
            addrField = new TextField();
            postalCodeField = new TextField();
            insertButton = new Button("Insert");
            searchButton = new Button("Back");

            // Set button styles
            insertButton.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white;");
            searchButton.setStyle("-fx-background-color: #3498db; -fx-text-fill: white;");

            currencyComboBox = new ComboBox<>();
            currencyComboBox.getItems().addAll("USD", "GBP", "EUR", "INR", "JPY"); // Add currency options
            currencyComboBox.setPromptText("Select Currency");
            
            // Set gender options
            genderComboBox = new ComboBox<>();
            genderComboBox.getItems().addAll("MALE", "FEMALE", "OTHER");
            genderComboBox.setPromptText("Select Gender");

            // Set age options
            ageComboBox = new ComboBox<>();
            for (double i = 1; i <= 100; i++) {
                ageComboBox.getItems().add((int) i);
            }
            ageComboBox.setPromptText("Select Age");

         // Set country code options
            countryCodeComboBox = new ComboBox<>();
            // Add country codes here
            countryCodeComboBox.getItems().addAll(
            	    "(Afghanistan) +93",
            	    "(Albania) +355",
            	    "(Algeria) +213",
            	    "(American Samoa) +1-684",
            	    "(Andorra) +376",
            	    "(Angola) +244",
            	    "(Anguilla) +1-264",
            	    "(Antarctica) +672",
            	    "(Antigua and Barbuda) +1-268",
            	    "(Argentina) +54",
            	    "(Armenia) +374",
            	    "(Aruba) +297",
            	    "(Australia) +61",
            	    "(Austria) +43",
            	    "(Azerbaijan) +994",
            	    "(Bahamas) +1-242",
            	    "(Bahrain) +973",
            	    "(Bangladesh) +880",
            	    "(Barbados) +1-246",
            	    "(Belarus) +375",
            	    "(Belgium) +32",
            	    "(Belize) +501",
            	    "(Benin) +229",
            	    "(Bermuda) +1-441",
            	    "(Bhutan) +975",
            	    "(Bolivia) +591",
            	    "(Bosnia and Herzegovina) +387",
            	    "(Botswana) +267",
            	    "(Brazil) +55",
            	    "(British Indian Ocean Territory) +246",
            	    "(British Virgin Islands) +1-284",
            	    "(Brunei) +673",
            	    "(Bulgaria) +359",
            	    "(Burkina Faso) +226",
            	    "(Burundi) +257",
            	    "(Cambodia) +855",
            	    "(Cameroon) +237",
            	    "(Canada) +1",
            	    "(Cape Verde) +238",
            	    "(Cayman Islands) +1-345",
            	    "(Central African Republic) +236",
            	    "(Chad) +235",
            	    "(Chile) +56",
            	    "(China) +86",
            	    "(Christmas Island) +61",
            	    "(Cocos Islands) +61",
            	    "(Colombia) +57",
            	    "(Comoros) +269",
            	    "(Cook Islands) +682",
            	    "(Costa Rica) +506",
            	    "(Croatia) +385",
            	    "(Cuba) +53",
            	    "(Curacao) +599",
            	    "(Cyprus) +357",
            	    "(Czech Republic) +420",
            	    "(Democratic Republic of the Congo) +243",
            	    "(Denmark) +45",
            	    "(Djibouti) +253",
            	    "(Dominica) +1-767",
            	    "(Dominican Republic) +1-809",
            	    "(East Timor) +670",
            	    "(Ecuador) +593",
            	    "(Egypt) +20",
            	    "(El Salvador) +503",
            	    "(Equatorial Guinea) +240",
            	    "(Eritrea) +291",
            	    "(Estonia) +372",
            	    "(Ethiopia) +251",
            	    "(Falkland Islands) +500",
            	    "(Faroe Islands) +298",
            	    "(Fiji) +679",
            	    "(Finland) +358",
            	    "(France) +33",
            	    "(French Polynesia) +689",
            	    "(Gabon) +241",
            	    "(Gambia) +220",
            	    "(Georgia) +995",
            	    "(Germany) +49",
            	    "(Ghana) +233",
            	    "(Gibraltar) +350",
            	    "(Greece) +30",
            	    "(Greenland) +299",
            	    "(Grenada) +1-473",
            	    "(Guam) +1-671",
            	    "(Guatemala) +502",
            	    "(Guernsey) +44-1481",
            	    "(Guinea) +224",
            	    "(Guinea-Bissau) +245",
            	    "(Guyana) +592",
            	    "(Haiti) +509",
            	    "(Honduras) +504",
            	    "(Hong Kong) +852",
            	    "(Hungary) +36",
            	    "(Iceland) +354",
            	    "(India) +91",
            	    "(Indonesia) +62",
            	    "(Iran) +98",
            	    "(Iraq) +964",
            	    "(Ireland) +353",
            	    "(Isle of Man) +44-1624",
            	    "(Israel) +972",
            	    "(Italy) +39",
            	    "(Ivory Coast) +225",
            	    "(Jamaica) +1-876",
            	    "(Japan) +81",
            	    "(Jersey) +44-1534",
            	    "(Jordan) +962",
            	    "(Kazakhstan) +7",
            	    "(Kenya) +254",
            	    "(Kiribati) +686",
            	    "(Kosovo) +383",
            	    "(Kuwait) +965",
            	    "(Kyrgyzstan) +996",
            	    "(Laos) +856",
            	    "(Latvia) +371",
            	    "(Lebanon) +961",
            	    "(Lesotho) +266",
            	    "(Liberia) +231",
            	    "(Libya) +218",
            	    "(Liechtenstein) +423",
            	    "(Lithuania) +370",
            	    "(Luxembourg) +352",
            	    "(Macau) +853",
            	    "(Macedonia) +389",
            	    "(Madagascar) +261",
            	    "(Malawi) +265",
            	    "(Malaysia) +60",
            	    "(Maldives) +960",
            	    "(Mali) +223",
            	    "(Malta) +356",
            	    "(Marshall Islands) +692",
            	    "(Mauritania) +222",
            	    "(Mauritius) +230",
            	    "(Mayotte) +262",
            	    "(Mexico) +52",
            	    "(Micronesia) +691",
            	    "(Moldova) +373",
            	    "(Monaco) +377",
            	    "(Mongolia) +976",
            	    "(Montenegro) +382",
            	    "(Montserrat) +1-664",
            	    "(Morocco) +212",
            	    "(Mozambique) +258",
            	    "(Myanmar) +95",
            	    "(Namibia) +264",
            	    "(Nauru) +674",
            	    "(Nepal) +977",
            	    "(Netherlands) +31",
            	    "(Netherlands Antilles) +599",
            	    "(New Caledonia) +687",
            	    "(New Zealand) +64",
            	    "(Nicaragua) +505",
            	    "(Niger) +227",
            	    "(Nigeria) +234",
            	    "(Niue) +683",
            	    "(North Korea) +850",
            	    "(Northern Mariana Islands) +1-670",
            	    "(Norway) +47",
            	    "(Oman) +968",
            	    "(Pakistan) +92",
            	    "(Palau) +680",
            	    "(Palestine) +970",
            	    "(Panama) +507",
            	    "(Papua New Guinea) +675",
            	    "(Paraguay) +595",
            	    "(Peru) +51",
            	    "(Philippines) +63",
            	    "(Pitcairn) +870",
            	    "(Poland) +48",
            	    "(Portugal) +351",
            	    "(Puerto Rico) +1-787",
            	    "(Puerto Rico) +1-939",
            	    "(Qatar) +974",
            	    "(Republic of the Congo) +242",
            	    "(Reunion) +262",
            	    "(Romania) +40",
            	    "(Russia) +7",
            	    "(Rwanda) +250",
            	    "(Saint Barthelemy) +590",
            	    "(Saint Helena) +290",
            	    "(Saint Kitts and Nevis) +1-869",
            	    "(Saint Lucia) +1-758",
            	    "(Saint Martin) +590",
            	    "(Saint Pierre and Miquelon) +508",
            	    "(Saint Vincent and the Grenadines) +1-784",
            	    "(Samoa) +685",
            	    "(San Marino) +378",
            	    "(Sao Tome and Principe) +239",
            	    "(Saudi Arabia) +966",
            	    "(Senegal) +221",
            	    "(Serbia) +381",
            	    "(Seychelles) +248",
            	    "(Sierra Leone) +232",
            	    "(Singapore) +65",
            	    "(Sint Maarten) +1-721",
            	    "(Slovakia) +421",
            	    "(Slovenia) +386",
            	    "(Solomon Islands) +677",
            	    "(Somalia) +252",
            	    "(South Africa) +27",
            	    "(South Korea) +82",
            	    "(South Sudan) +211",
            	    "(Spain) +34",
            	    "(Sri Lanka) +94",
            	    "(Sudan) +249",
            	    "(Suriname) +597",
            	    "(Svalbard and Jan Mayen) +47",
            	    "(Swaziland) +268",
            	    "(Sweden) +46",
            	    "(Switzerland) +41",
            	    "(Syria) +963",
            	    "(Taiwan) +886",
            	    "(Tajikistan) +992",
            	    "(Tanzania) +255",
            	    "(Thailand) +66",
            	    "(Togo) +228",
            	    "(Tokelau) +690",
            	    "(Tonga) +676",
            	    "(Trinidad and Tobago) +1-868",
            	    "(Tunisia) +216",
            	    "(Turkey) +90",
            	    "(Turkmenistan) +993",
            	    "(Turks and Caicos Islands) +1-649",
            	    "(Tuvalu) +688",
            	    "(U.S. Virgin Islands) +1-340",
            	    "(Uganda) +256",
            	    "(Ukraine) +380",
            	    "(United Arab Emirates) +971",
            	    "(United Kingdom) +44",
            	    "(United States) +1",
            	    "(Uruguay) +598",
            	    "(Uzbekistan) +998",
            	    "(Vanuatu) +678",
            	    "(Vatican) +379",
            	    "(Venezuela) +58",
            	    "(Vietnam) +84",
            	    "(Wallis and Futuna) +681",
            	    "(Western Sahara) +212",
            	    "(Yemen) +967",
            	    "(Zambia) +260",
            	    "(Zimbabwe) +263"
            	);


            countryCodeComboBox.setPromptText("Select Country Code");
            
            // Set layout
            GridPane inputGridPane = new GridPane();
            inputGridPane.setVgap(10);
            inputGridPane.setHgap(10);
            inputGridPane.addRow(0, new Label("ID:"), idField);
            inputGridPane.addRow(1, new Label("First Name:"), firstNameField);
            inputGridPane.addRow(2, new Label("Middle Name:"), middleNameField);
            inputGridPane.addRow(3, new Label("Last Name:"), lastNameField);
            inputGridPane.addRow(4, new Label("Position:"), positionField);
            inputGridPane.addRow(5, new Label("Salary:"), new HBox(currencyComboBox, salaryField));
            inputGridPane.addRow(6, new Label("Age:"), ageComboBox); // Add ComboBox for age
            inputGridPane.addRow(7, new Label("Gender:"), genderComboBox); // Add ComboBox for gender
            inputGridPane.addRow(8, new Label("Phone Number:"), new HBox(countryCodeComboBox, phnoField)); // Add ComboBox for country code
            inputGridPane.addRow(9, new Label("Address:"), addrField);
            inputGridPane.addRow(10, new Label("Email:"), postalCodeField); // Changed Postal Code to Email
            inputGridPane.add(insertButton, 0, 11);
            inputGridPane.add(searchButton, 1, 11);

            // Add action listeners
            insertButton.setOnAction(e -> insertData());
            searchButton.setOnAction(e -> {
                primaryStage.close();
                new tast().start(new Stage());
            });

            // Set scene
            Scene scene = new Scene(inputGridPane, 800, 600);
            primaryStage.setScene(scene);
            primaryStage.show();
        }

        private void insertData() {
        	double salary = Double.parseDouble(salaryField.getText());
        	String salaryString = " " + String.valueOf(salary);
            String currency = currencyComboBox.getValue(); // Get selected currency from ComboBox
            String currencysalary = currency.concat(salaryString);
            // Validate email format
            String email = postalCodeField.getText();
            if (!isValidEmail(email)) {
                showAlert("Invalid email address!");
                return; // Exit method if email is invalid
            }

            // Insert data into the database
            try (Connection conn = DriverManager.getConnection("jdbc:ucanaccess://C:/Users/syedr/OneDrive/Desktop/rayan/Projects/Employee Database Management/Database21.accdb");
                 PreparedStatement pstmt = conn.prepareStatement("INSERT INTO Employees (id, first_name, middle_name, last_name, position, salary, age, gender, phno, addr, Email) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")) {

                int id = Integer.parseInt(idField.getText());
                String firstName = firstNameField.getText();
                String middleName = middleNameField.getText();
                String lastName = lastNameField.getText();
                String position = positionField.getText();
                double age = ageComboBox.getValue(); // Get selected age from ComboBox
                String gender = genderComboBox.getValue(); // Get selected gender from ComboBox
                String phno1 = countryCodeComboBox.getValue();
                String phno2= " " + phnoField.getText();
                String phno = phno1.concat(phno2);
                String addr = addrField.getText();
                String emailaddr = postalCodeField.getText(); // Changed Postal Code to Email
                
                
                pstmt.setInt(1, id);
                pstmt.setString(2, firstName);
                pstmt.setString(3, middleName);
                pstmt.setString(4, lastName);
                pstmt.setString(5, position);
                pstmt.setString(6, currencysalary);
                pstmt.setDouble(7, age);
                pstmt.setString(8, gender); // Set gender
                pstmt.setString(9, phno);
                pstmt.setString(10, addr);
                pstmt.setString(11, emailaddr); // Changed Postal Code to Email

                int rowsAffected = pstmt.executeUpdate();
                showAlert(rowsAffected + " row(s) inserted successfully.");
            } catch (SQLException ex) {
                showAlert("Error inserting data: " + ex.getMessage());
            }
        }

        // Validate email using regular expression
        private boolean isValidEmail(String email) {
            String emailRegex = "^[a-zA-Z0-9_+&*-]+(?:\\.[a-zA-Z0-9_+&*-]+)*@(?:[a-zA-Z0-9-]+\\.)+[a-zA-Z]{2,7}$";
            return email.matches(emailRegex);
        }

        // Helper method to show alert messages
        private void showAlert(String message) {
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Information");
            alert.setHeaderText(null);
            alert.setContentText(message);
            alert.showAndWait();
        }
    }

    // Method to print all data in TableView format
    @SuppressWarnings("unchecked")
    private void printAllData() {
        try (Connection conn = DriverManager.getConnection("jdbc:ucanaccess://C:/Users/syedr/OneDrive/Desktop/rayan/Projects/Employee Database Management/Database21.accdb");
             Statement stmt = conn.createStatement()) {

            ResultSet rs = stmt.executeQuery("SELECT * FROM Employees");

            // Create TableView and columns
            TableView<Employee> tableView = new TableView<>();

            // Define columns for all fields
            TableColumn<Employee, Integer> idColumn = new TableColumn<>("ID");
            TableColumn<Employee, String> firstNameColumn = new TableColumn<>("First Name");
            TableColumn<Employee, String> middleNameColumn = new TableColumn<>("Middle Name");
            TableColumn<Employee, String> lastNameColumn = new TableColumn<>("Last Name");
            TableColumn<Employee, String> positionColumn = new TableColumn<>("Position");
            TableColumn<Employee, Double> salaryColumn = new TableColumn<>("Salary");
            TableColumn<Employee, Integer> ageColumn = new TableColumn<>("Age");
            TableColumn<Employee, String> genderColumn = new TableColumn<>("Gender");
            TableColumn<Employee, String> phnoColumn = new TableColumn<>("Phone Number");
            TableColumn<Employee, String> addrColumn = new TableColumn<>("Address");
            TableColumn<Employee, String> emailColumn = new TableColumn<>("Email"); // Changed Postal Code to Email

            // Set cell value factories for each column
            idColumn.setCellValueFactory(new PropertyValueFactory<>("id"));
            firstNameColumn.setCellValueFactory(new PropertyValueFactory<>("firstName"));
            middleNameColumn.setCellValueFactory(new PropertyValueFactory<>("middleName"));
            lastNameColumn.setCellValueFactory(new PropertyValueFactory<>("lastName"));
            positionColumn.setCellValueFactory(new PropertyValueFactory<>("position"));
            salaryColumn.setCellValueFactory(new PropertyValueFactory<>("salary"));
            ageColumn.setCellValueFactory(new PropertyValueFactory<>("age"));
            genderColumn.setCellValueFactory(new PropertyValueFactory<>("gender"));
            phnoColumn.setCellValueFactory(new PropertyValueFactory<>("phno"));
            addrColumn.setCellValueFactory(new PropertyValueFactory<>("addr"));
            emailColumn.setCellValueFactory(new PropertyValueFactory<>("email")); // Changed Postal Code to Email

            // Add columns to TableView
            tableView.getColumns().addAll(idColumn, firstNameColumn, middleNameColumn, lastNameColumn,
                                            positionColumn, salaryColumn, ageColumn, genderColumn,
                                            phnoColumn, addrColumn, emailColumn); // Changed Postal Code to Email

            // Populate TableView with data
            ObservableList<Employee> data = FXCollections.observableArrayList();
            while (rs.next()) {
                data.add(new Employee(
                        rs.getInt("id"),
                        rs.getString("first_name"),
                        rs.getString("middle_name"),
                        rs.getString("last_name"),
                        rs.getString("position"),
                        rs.getString("salary"),
                        rs.getDouble("age"),
                        rs.getString("gender"),
                        rs.getString("phno"),
                        rs.getString("addr"),
                        rs.getString("Email") // Changed Postal Code to Email
                ));
            }
            tableView.setItems(data);

            // Create a new stage to display the TableView
            Stage tableViewStage = new Stage();
            tableViewStage.setTitle("Employee Data");
            tableViewStage.setScene(new Scene(tableView, 800, 600));
            tableViewStage.show();

        } catch (SQLException ex) {
            showAlert("Error printing data: " + ex.getMessage());
        }
    }


    

    // Helper method to show alert messages
    private void showAlert(String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Information");
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}

class ViewFrame extends Application {
	private TextField idField;
	    private TextArea outputArea;
	    private Button searchButton, backButton;

	    @Override
	    public void start(Stage primaryStage) {
	        primaryStage.setTitle("View Data");

	        // Initialize components
	        idField = new TextField();
	        outputArea = new TextArea();
	        outputArea.setEditable(false); // Make it read-only
	        searchButton = new Button("Search");
	        backButton = new Button("Back");

	        // Set button styles
	        searchButton.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white;");
	        backButton.setStyle("-fx-background-color: #3498db; -fx-text-fill: white;");

	        // Set layout
	        VBox layout = new VBox(10);
	        layout.setStyle("-fx-background-color: #ecf0f1; -fx-padding: 10px;");
	        layout.getChildren().addAll(
	                new HBox(new Label("Enter Employee ID:"), idField),
	                searchButton,
	                backButton,
	                outputArea
	        );

	        // Add action listeners
	        searchButton.setOnAction(e -> fetchData(Integer.parseInt(idField.getText())));
	        backButton.setOnAction(e -> {
	            primaryStage.close();
	            new tast().start(new Stage());
	        });

	        // Set scene
	        Scene scene = new Scene(layout, 800, 600);
	        primaryStage.setScene(scene);
	        primaryStage.show();
	    }

	    private void fetchData(int id) {
	        // Fetch data from the database for the specified employee ID
	        try (Connection conn = DriverManager.getConnection("jdbc:ucanaccess://C:/Users/syedr/OneDrive/Desktop/rayan/Projects/Employee Database Management/Database21.accdb");
	             PreparedStatement pstmt = conn.prepareStatement("SELECT * FROM Employees WHERE id=?")) {

	            pstmt.setInt(1, id);
	            ResultSet rs = pstmt.executeQuery();

	            if (rs.next()) {
	                String firstName = rs.getString("first_name");
	                String middleName = rs.getString("middle_name");
	                String lastName = rs.getString("last_name");
	                String position = rs.getString("position");
	                String salary = rs.getString("salary");
	                double age = rs.getInt("age");
	                String gender = rs.getString("gender");
	                String phno = rs.getString("phno");
	                String addr = rs.getString("addr");
	                String emailAddress = rs.getString("Email");

	                outputArea.setText("ID: " + id + "\n" +
	                        "First Name: " + firstName + "\n" +
	                        "Middle Name: " + middleName + "\n" +
	                        "Last Name: " + lastName + "\n" +
	                        "Position: " + position + "\n" +
	                        "Salary: " + salary + "\n" +
	                        "Age: " + age + "\n" +
	                        "Gender: " + gender + "\n" +
	                        "Phone Number: " + phno + "\n" +
	                        "Address: " + addr + "\n" +
	                        "Email Address: " + emailAddress + "\n");
	            } else {
	                showAlert("Employee with ID " + id + " not found");
	            }
	        } catch (SQLException ex) {
	            showAlert("Error fetching data: " + ex.getMessage());
	        }
	    }

	    // Helper method to show alert messages
	    private void showAlert(String message) {
	        Alert alert = new Alert(Alert.AlertType.INFORMATION);
	        alert.setTitle("Information");
	        alert.setHeaderText(null);
	        alert.setContentText(message);
	        alert.showAndWait();
	    }
	}

class UpdateFrame extends Application {
	   private TextField idField, firstNameField, middleNameField, lastNameField, positionField, salaryField, phnoField, addrField, emailField;
       private ComboBox<String> currencyComboBox; // Add ComboBox for currency selection
	    private ComboBox<Double> ageComboBox; // ComboBox for age selection
	    private ComboBox<String> genderComboBox; // ComboBox for gender selection
	    private ComboBox<String> phnoRegionalCodeComboBox; // ComboBox for phone number regional code selection
	    private Button updateButton, backButton;

	    @Override
	    public void start(Stage primaryStage) {
	        primaryStage.setTitle("Update Data");

	        // Initialize components
	        idField = new TextField();
	        firstNameField = new TextField();
	        middleNameField = new TextField();
	        lastNameField = new TextField();
	        positionField = new TextField();
	        salaryField = new TextField();
	        phnoField = new TextField();
	        addrField = new TextField();
	        emailField = new TextField();
	        updateButton = new Button("Update");
	        backButton = new Button("Back");

	        // Set button styles
	        updateButton.setStyle("-fx-background-color: #f39c12; -fx-text-fill: white;");
	        backButton.setStyle("-fx-background-color: #3498db; -fx-text-fill: white;");
	        
            currencyComboBox = new ComboBox<>();
            currencyComboBox.getItems().addAll("USD", "GBP", "EUR", "INR", "JPY"); // Add currency options
            currencyComboBox.setPromptText("Select Currency");
            
	        // Set gender options
	        genderComboBox = new ComboBox<>();
	        genderComboBox.getItems().addAll("MALE", "FEMALE", "OTHER");
	        genderComboBox.setPromptText("Select Gender");

	        // Set age options
	        ageComboBox = new ComboBox<>();
	        for (double i = 1; i <= 100; i++) {
	            ageComboBox.getItems().add(i);
	        }
	        ageComboBox.setPromptText("Select Age");
	        
	        // Set phone number regional code options
	        phnoRegionalCodeComboBox = new ComboBox<>();
	        // Add regional codes of countries here
	        phnoRegionalCodeComboBox.getItems().addAll(
	        		"(Afghanistan) +93",
	        	    "(Albania) +355",
	        	    "(Algeria) +213",
	        	    "(American Samoa) +1-684",
	        	    "(Andorra) +376",
	        	    "(Angola) +244",
	        	    "(Anguilla) +1-264",
	        	    "(Antarctica) +672",
	        	    "(Antigua and Barbuda) +1-268",
	        	    "(Argentina) +54",
	        	    "(Armenia) +374",
	        	    "(Aruba) +297",
	        	    "(Australia) +61",
	        	    "(Austria) +43",
	        	    "(Azerbaijan) +994",
	        	    "(Bahamas) +1-242",
	        	    "(Bahrain) +973",
	        	    "(Bangladesh) +880",
	        	    "(Barbados) +1-246",
	        	    "(Belarus) +375",
	        	    "(Belgium) +32",
	        	    "(Belize) +501",
	        	    "(Benin) +229",
	        	    "(Bermuda) +1-441",
	        	    "(Bhutan) +975",
	        	    "(Bolivia) +591",
	        	    "(Bosnia and Herzegovina) +387",
	        	    "(Botswana) +267",
	        	    "(Brazil) +55",
	        	    "(British Indian Ocean Territory) +246",
	        	    "(British Virgin Islands) +1-284",
	        	    "(Brunei) +673",
	        	    "(Bulgaria) +359",
	        	    "(Burkina Faso) +226",
	        	    "(Burundi) +257",
	        	    "(Cambodia) +855",
	        	    "(Cameroon) +237",
	        	    "(Canada) +1",
	        	    "(Cape Verde) +238",
	        	    "(Cayman Islands) +1-345",
	        	    "(Central African Republic) +236",
	        	    "(Chad) +235",
	        	    "(Chile) +56",
	        	    "(China) +86",
	        	    "(Christmas Island) +61",
	        	    "(Cocos Islands) +61",
	        	    "(Colombia) +57",
	        	    "(Comoros) +269",
	        	    "(Cook Islands) +682",
	        	    "(Costa Rica) +506",
	        	    "(Croatia) +385",
	        	    "(Cuba) +53",
	        	    "(Curacao) +599",
	        	    "(Cyprus) +357",
	        	    "(Czech Republic) +420",
	        	    "(Democratic Republic of the Congo) +243",
	        	    "(Denmark) +45",
	        	    "(Djibouti) +253",
	        	    "(Dominica) +1-767",
	        	    "(Dominican Republic) +1-809",
	        	    "(East Timor) +670",
	        	    "(Ecuador) +593",
	        	    "(Egypt) +20",
	        	    "(El Salvador) +503",
	        	    "(Equatorial Guinea) +240",
	        	    "(Eritrea) +291",
	        	    "(Estonia) +372",
	        	    "(Ethiopia) +251",
	        	    "(Falkland Islands) +500",
	        	    "(Faroe Islands) +298",
	        	    "(Fiji) +679",
	        	    "(Finland) +358",
	        	    "(France) +33",
	        	    "(French Polynesia) +689",
	        	    "(Gabon) +241",
	        	    "(Gambia) +220",
	        	    "(Georgia) +995",
	        	    "(Germany) +49",
	        	    "(Ghana) +233",
	        	    "(Gibraltar) +350",
	        	    "(Greece) +30",
	        	    "(Greenland) +299",
	        	    "(Grenada) +1-473",
	        	    "(Guam) +1-671",
	        	    "(Guatemala) +502",
	        	    "(Guernsey) +44-1481",
	        	    "(Guinea) +224",
	        	    "(Guinea-Bissau) +245",
	        	    "(Guyana) +592",
	        	    "(Haiti) +509",
	        	    "(Honduras) +504",
	        	    "(Hong Kong) +852",
	        	    "(Hungary) +36",
	        	    "(Iceland) +354",
	        	    "(India) +91",
	        	    "(Indonesia) +62",
	        	    "(Iran) +98",
	        	    "(Iraq) +964",
	        	    "(Ireland) +353",
	        	    "(Isle of Man) +44-1624",
	        	    "(Israel) +972",
	        	    "(Italy) +39",
	        	    "(Ivory Coast) +225",
	        	    "(Jamaica) +1-876",
	        	    "(Japan) +81",
	        	    "(Jersey) +44-1534",
	        	    "(Jordan) +962",
	        	    "(Kazakhstan) +7",
	        	    "(Kenya) +254",
	        	    "(Kiribati) +686",
	        	    "(Kosovo) +383",
	        	    "(Kuwait) +965",
	        	    "(Kyrgyzstan) +996",
	        	    "(Laos) +856",
	        	    "(Latvia) +371",
	        	    "(Lebanon) +961",
	        	    "(Lesotho) +266",
	        	    "(Liberia) +231",
	        	    "(Libya) +218",
	        	    "(Liechtenstein) +423",
	        	    "(Lithuania) +370",
	        	    "(Luxembourg) +352",
	        	    "(Macau) +853",
	        	    "(Macedonia) +389",
	        	    "(Madagascar) +261",
	        	    "(Malawi) +265",
	        	    "(Malaysia) +60",
	        	    "(Maldives) +960",
	        	    "(Mali) +223",
	        	    "(Malta) +356",
	        	    "(Marshall Islands) +692",
	        	    "(Mauritania) +222",
	        	    "(Mauritius) +230",
	        	    "(Mayotte) +262",
	        	    "(Mexico) +52",
	        	    "(Micronesia) +691",
	        	    "(Moldova) +373",
	        	    "(Monaco) +377",
	        	    "(Mongolia) +976",
	        	    "(Montenegro) +382",
	        	    "(Montserrat) +1-664",
	        	    "(Morocco) +212",
	        	    "(Mozambique) +258",
	        	    "(Myanmar) +95",
	        	    "(Namibia) +264",
	        	    "(Nauru) +674",
	        	    "(Nepal) +977",
	        	    "(Netherlands) +31",
	        	    "(Netherlands Antilles) +599",
	        	    "(New Caledonia) +687",
	        	    "(New Zealand) +64",
	        	    "(Nicaragua) +505",
	        	    "(Niger) +227",
	        	    "(Nigeria) +234",
	        	    "(Niue) +683",
	        	    "(North Korea) +850",
	        	    "(Northern Mariana Islands) +1-670",
	        	    "(Norway) +47",
	        	    "(Oman) +968",
	        	    "(Pakistan) +92",
	        	    "(Palau) +680",
	        	    "(Palestine) +970",
	        	    "(Panama) +507",
	        	    "(Papua New Guinea) +675",
	        	    "(Paraguay) +595",
	        	    "(Peru) +51",
	        	    "(Philippines) +63",
	        	    "(Pitcairn) +870",
	        	    "(Poland) +48",
	        	    "(Portugal) +351",
	        	    "(Puerto Rico) +1-787",
	        	    "(Puerto Rico) +1-939",
	        	    "(Qatar) +974",
	        	    "(Republic of the Congo) +242",
	        	    "(Reunion) +262",
	        	    "(Romania) +40",
	        	    "(Russia) +7",
	        	    "(Rwanda) +250",
	        	    "(Saint Barthelemy) +590",
	        	    "(Saint Helena) +290",
	        	    "(Saint Kitts and Nevis) +1-869",
	        	    "(Saint Lucia) +1-758",
	        	    "(Saint Martin) +590",
	        	    "(Saint Pierre and Miquelon) +508",
	        	    "(Saint Vincent and the Grenadines) +1-784",
	        	    "(Samoa) +685",
	        	    "(San Marino) +378",
	        	    "(Sao Tome and Principe) +239",
	        	    "(Saudi Arabia) +966",
	        	    "(Senegal) +221",
	        	    "(Serbia) +381",
	        	    "(Seychelles) +248",
	        	    "(Sierra Leone) +232",
	        	    "(Singapore) +65",
	        	    "(Sint Maarten) +1-721",
	        	    "(Slovakia) +421",
	        	    "(Slovenia) +386",
	        	    "(Solomon Islands) +677",
	        	    "(Somalia) +252",
	        	    "(South Africa) +27",
	        	    "(South Korea) +82",
	        	    "(South Sudan) +211",
	        	    "(Spain) +34",
	        	    "(Sri Lanka) +94",
	        	    "(Sudan) +249",
	        	    "(Suriname) +597",
	        	    "(Svalbard and Jan Mayen) +47",
	        	    "(Swaziland) +268",
	        	    "(Sweden) +46",
	        	    "(Switzerland) +41",
	        	    "(Syria) +963",
	        	    "(Taiwan) +886",
	        	    "(Tajikistan) +992",
	        	    "(Tanzania) +255",
	        	    "(Thailand) +66",
	        	    "(Togo) +228",
	        	    "(Tokelau) +690",
	        	    "(Tonga) +676",
	        	    "(Trinidad and Tobago) +1-868",
	        	    "(Tunisia) +216",
	        	    "(Turkey) +90",
	        	    "(Turkmenistan) +993",
	        	    "(Turks and Caicos Islands) +1-649",
	        	    "(Tuvalu) +688",
	        	    "(U.S. Virgin Islands) +1-340",
	        	    "(Uganda) +256",
	        	    "(Ukraine) +380",
	        	    "(United Arab Emirates) +971",
	        	    "(United Kingdom) +44",
	        	    "(United States) +1",
	        	    "(Uruguay) +598",
	        	    "(Uzbekistan) +998",
	        	    "(Vanuatu) +678",
	        	    "(Vatican) +379",
	        	    "(Venezuela) +58",
	        	    "(Vietnam) +84",
	        	    "(Wallis and Futuna) +681",
	        	    "(Western Sahara) +212",
	        	    "(Yemen) +967",
	        	    "(Zambia) +260",
	        	    "(Zimbabwe) +263");
	        phnoRegionalCodeComboBox.setPromptText("Select Regional Code");

	        // Set layout
	        GridPane inputGridPane = new GridPane();
	        inputGridPane.setVgap(10);
	        inputGridPane.setHgap(10);
	        inputGridPane.addRow(0, new Label("ID:"), idField);
	        inputGridPane.addRow(1, new Label("First Name:"), firstNameField);
	        inputGridPane.addRow(2, new Label("Middle Name:"), middleNameField);
	        inputGridPane.addRow(3, new Label("Last Name:"), lastNameField);
	        inputGridPane.addRow(4, new Label("Position:"), positionField);
            inputGridPane.addRow(5, new Label("Salary:"), new HBox(currencyComboBox, salaryField));
	        inputGridPane.addRow(6, new Label("Age:"), ageComboBox); // Add ComboBox for age
	        inputGridPane.addRow(7, new Label("Gender:"), genderComboBox); // Add ComboBox for gender
	        inputGridPane.addRow(8, new Label("Phone Number:"), new HBox(phnoRegionalCodeComboBox, phnoField));
	        inputGridPane.addRow(9, new Label("Address:"), addrField);
	        inputGridPane.addRow(10, new Label("Email Address:"), emailField);
	        inputGridPane.add(updateButton, 0, 11);
	        inputGridPane.add(backButton, 1, 11);

	        // Add action listeners
	        updateButton.setOnAction(e -> updateData());
	        backButton.setOnAction(e -> primaryStage.close());

	        // Set scene
	        Scene scene = new Scene(inputGridPane, 800, 600);
	        primaryStage.setScene(scene);
	        primaryStage.show();
	    }

	    private void updateData() {
	    	
	    	double salary = Double.parseDouble(salaryField.getText());
	    	String salaryString = " " + String.valueOf(salary);
	    	String currency = currencyComboBox.getValue(); // Get selected currency from ComboBox
	    	String currencysalary = currency.concat(salaryString);
	    	// Validate email format
	        String email = emailField.getText();
	        if (!isValidEmail(email)) {
	            showAlert("Invalid email address!");
	            return; // Exit method if email is invalid
	        }

	        // Update data in the database
	        try (Connection conn = DriverManager.getConnection("jdbc:ucanaccess://C:/Users/syedr/OneDrive/Desktop/rayan/Projects/Employee Database Management/Database21.accdb");
	             PreparedStatement pstmt = conn.prepareStatement("UPDATE Employees SET first_name=?, middle_name=?, last_name=?, position=?, salary=?, age=?, gender=?, phno=?, addr=?, Email=? WHERE id=?")) {

	            int id = Integer.parseInt(idField.getText());
	            String firstName = firstNameField.getText();
	            String middleName = middleNameField.getText();
	            String lastName = lastNameField.getText();
	            String position = positionField.getText();
	            double age = ageComboBox.getValue(); // Get selected age from ComboBox
	            String gender = genderComboBox.getValue(); // Get selected gender from ComboBo
                String phno1 = phnoRegionalCodeComboBox.getValue();
                String phno2= " " + phnoField.getText();
                String phno = phno1.concat(phno2);
	            String addr = addrField.getText();
	            String emailAddress = emailField.getText();

	            pstmt.setString(1, firstName);
	            pstmt.setString(2, middleName);
	            pstmt.setString(3, lastName);
	            pstmt.setString(4, position);
	            pstmt.setString(5, currencysalary);
	            pstmt.setDouble(6, age);
	            pstmt.setString(7, gender); // Set gender
	            pstmt.setString(8, phno);
	            pstmt.setString(9, addr);
	            pstmt.setString(10, emailAddress);
	            pstmt.setInt(11, id);

	            int rowsAffected = pstmt.executeUpdate();
	            showAlert(rowsAffected + " row(s) updated successfully.");
	        } catch (SQLException ex) {
	            showAlert("Error updating data: " + ex.getMessage());
	        }
	    }

	    private boolean isValidEmail(String email) {
	        // Regular expression for email validation
	        String emailRegex = "^[a-zA-Z0-9_+&*-]+(?:\\.[a-zA-Z0-9_+&*-]+)*@(?:[a-zA-Z0-9-]+\\.)+[a-zA-Z]{2,7}$";
	        return email.matches(emailRegex);
	    }


		// Helper method to show alert messages
	    private void showAlert(String message) {
	        Alert alert = new Alert(Alert.AlertType.INFORMATION);
	        alert.setTitle("Information");
	        alert.setHeaderText(null);
	        alert.setContentText(message);
	        alert.showAndWait();
	    }
	}