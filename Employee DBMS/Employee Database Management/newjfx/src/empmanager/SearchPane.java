package empmanager;

/**Classes needed to build the Menu */
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;

public class SearchPane extends BorderPane {

    protected final ObservableList<Employee> data = FXCollections.observableArrayList();

    /** Create a menu bar */
    private MenuBar menuBar = new MenuBar();
    protected Menu mainMenu = new Menu("Main menu");
    protected MenuItem addEmployee = new MenuItem("Add Employee");
    protected MenuItem deleteEmployee = new MenuItem("Delete Employee");
    protected MenuItem updateEmployee = new MenuItem("Update Employee");
    protected MenuItem help = new MenuItem("Help");
    protected MenuItem exit = new MenuItem("Exit");

    /** List All button */
    protected Button listAllButton = new Button("List all employees in the System");

    /** Create a combo box */
    protected ComboBox<String> comboBoxSearch = new ComboBox<>();

    /** Create a search field */
    protected TextField searchField = new TextField();

    /** Create table view and table columns */
    protected TableView<Employee> table = new TableView<>();
    protected TableColumn<Employee, Integer> idCol = new TableColumn<>("Id");
    protected TableColumn<Employee, String> firstNameCol = new TableColumn<>("First name");
    protected TableColumn<Employee, String> lastNameCol = new TableColumn<>("Last name");
    protected TableColumn<Employee, String> middleNameCol = new TableColumn<>("Middle name");
    protected TableColumn<Employee, String> positionCol = new TableColumn<>("Position");
    protected TableColumn<Employee, Double> salaryCol = new TableColumn<>("Salary");

    /** Create title label */
    private Label searchLabel = new Label("Search");

    /** Create a VBox */
    private HBox searchBox = new HBox();
    private VBox vbox = new VBox();

    /** Default constructor */
    @SuppressWarnings("unchecked")
	public SearchPane() {
        /** Set the menu properties */
        menuBar.setStyle("-fx-background-color: rgb(205,205,205);");
        mainMenu.setStyle("-fx-font-size: 12;" + "-fx-font-family: calibri;");

        menuBar.getMenus().add(mainMenu);
        mainMenu.getItems().addAll(addEmployee, deleteEmployee, updateEmployee, help, exit);

        searchLabel.setStyle("-fx-text-fill: black;" + "-fx-font-family: Arial;" + "-fx-font-size: 13;");

        /** Set the table view properties */
        table.getColumns().addAll(idCol, firstNameCol, middleNameCol, lastNameCol, positionCol, salaryCol);
        idCol.setMinWidth(50);
        firstNameCol.setMinWidth(100);
        middleNameCol.setMinWidth(100);
        lastNameCol.setMinWidth(100);
        positionCol.setMinWidth(100);
        salaryCol.setMinWidth(150);

        idCol.setCellValueFactory(new PropertyValueFactory<>("id"));
        firstNameCol.setCellValueFactory(new PropertyValueFactory<>("firstName"));
        middleNameCol.setCellValueFactory(new PropertyValueFactory<>("middleName"));
        lastNameCol.setCellValueFactory(new PropertyValueFactory<>("lastName"));
        positionCol.setCellValueFactory(new PropertyValueFactory<>("position"));
        salaryCol.setCellValueFactory(new PropertyValueFactory<>("salary"));

        table.setItems(data);

        /** Set the search field properties */
        searchField.setStyle("-fx-background: yellow;");
        searchField.setPrefWidth(200);

        /** HBox properties */
        searchBox.setSpacing(10);
        searchBox.setPadding(new Insets(10, 100, 0, 0));

        searchBox.getChildren().addAll(comboBoxSearch, searchField, listAllButton);

        /** Set combo box properties */
        comboBoxSearch.getItems().addAll("Id", "First name", "Middle name", "Last name", "Position", "Salary");
        comboBoxSearch.setPrefWidth(150);
        comboBoxSearch.setValue("Id");

        /** VBox properties */
        vbox.setSpacing(10);
        vbox.setPadding(new Insets(10, 10, 10, 10));
        vbox.getChildren().addAll(searchLabel, searchBox, table);

        /** Border pane */
        setTop(menuBar);
        setCenter(vbox);
    }
}
