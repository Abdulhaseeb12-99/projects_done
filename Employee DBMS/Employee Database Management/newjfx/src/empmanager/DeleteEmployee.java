package empmanager;

import java.sql.Statement;
import java.util.ArrayList;
import javax.swing.JOptionPane;

import javafx.collections.ObservableList;
import javafx.geometry.HPos;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.stage.Modality;
import javafx.stage.Stage;

public class DeleteEmployee {

    private TextField tfId = new TextField();
    private Label idLabel = new Label("Id");
    private Label titleLabel = new Label("Delete employee: ");
    private Button delete = new Button("Delete");
    private Button cancel = new Button("Cancel");
    private VBox vbox = new VBox();
    private HBox hbox = new HBox();
    private GridPane gridPane = new GridPane();

    public DeleteEmployee(Stage primaryStage, Statement stm, ObservableList<Employee> data, ArrayList<Employee> employees) {
        tfId.setStyle("-fx-background:yellow;");
        tfId.setPrefWidth(200);

        idLabel.setStyle("-fx-text-fill: black;" + "-fx-font-family:calibri;" + "-fx-font-size:14;");
        titleLabel.setStyle("-fx-text-fill: black;" + "-fx-font-family:calibri;" + "-fx-font-size:16;");

        delete.setPrefWidth(100);
        cancel.setPrefWidth(100);

        hbox.setSpacing(10);
        hbox.getChildren().addAll(delete, cancel);

        gridPane.setAlignment(Pos.CENTER);
        gridPane.setHgap(15);
        gridPane.setVgap(20);
        gridPane.setPadding(new Insets(10, 20, 0, 0));
        gridPane.add(idLabel, 0, 0);
        gridPane.add(tfId, 1, 0);
        gridPane.add(hbox, 1, 1);
        GridPane.setHalignment(hbox, HPos.RIGHT);

        vbox.setSpacing(25);
        vbox.setPadding(new Insets(10, 20, 0, 0));
        vbox.getChildren().addAll(titleLabel, gridPane);

        Scene scene = new Scene(vbox, 300, 150);
        Stage mainStage = new Stage();
        mainStage.initModality(Modality.APPLICATION_MODAL);
        mainStage.initOwner(primaryStage);

        mainStage.setScene(scene);
        mainStage.setTitle("Employee Manager App (Delete employee)");
        mainStage.show();

        cancel.setOnAction(e -> mainStage.close());

        delete.setOnAction(e -> {
            try {
                deleteEmployee(mainStage, stm, data, employees);
            } catch (Exception exception) {
                popErrorMessage(mainStage, exception.getMessage());
            }
        });

        vbox.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                try {
                    deleteEmployee(primaryStage, stm, data, employees);
                } catch (Exception exception) {
                    popErrorMessage(mainStage, exception.getMessage());
                }
            }
        });
    }

    private void deleteEmployee(Stage stage, Statement stm, ObservableList<Employee> data, ArrayList<Employee> employees) throws Exception {
        String idText = tfId.getText();
        if (idText.isEmpty()) {
            popErrorMessage(stage, "ID field cannot be empty!");
            return;
        }

        int id;
        try {
            id = Integer.parseInt(idText);
        } catch (NumberFormatException e) {
            popErrorMessage(stage, "Invalid ID format!");
            return;
        }

        String query = "DELETE FROM employee WHERE id='" + id + "'";

        int result = JOptionPane.showOptionDialog(null, "Are you sure you want to delete this record", "Delete Record",
                JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE, null, null, null);
        if (result == JOptionPane.YES_OPTION) {
            boolean employeeFound = false;
            for (int i = 0; i < data.size(); i++) {
                if (data.get(i).getId() == id) {
                    stm.execute(query);
                    data.remove(i);
                    employees.remove(i);
                    employeeFound = true;
                    break;
                }
            }

            if (employeeFound) {
                tfId.clear();
                infoDialog(stage, "Employee was successfully deleted!");
            } else {
                popErrorMessage(stage, "Employee doesn't exist!");
            }
        }
    }

    private void popErrorMessage(Stage primaryStage, String message) {
        final Stage dialog = new Stage();
        dialog.initModality(Modality.APPLICATION_MODAL);
        dialog.initOwner(primaryStage);

        Text text = new Text(message);
        text.setFill(Color.RED);
        text.setFont(Font.font("Times New Roman", 20));

        ImageView image = new ImageView(new Image("file:warning.png"));

        Button dialogOk = new Button("OK");
        dialogOk.setPrefWidth(100);

        HBox dialogHbox = new HBox(10, image, text);
        HBox.setMargin(image, new Insets(0, 0, 0, 10));

        VBox dialogVBox = new VBox(10, dialogHbox, dialogOk);
        VBox.setMargin(dialogOk, new Insets(0, 0, 0, 170));

        dialogOk.setOnAction(e -> dialog.close());

        Scene scene = new Scene(dialogVBox, 450, 100);

        dialog.setScene(scene);
        dialog.setTitle("Error Message");
        dialog.show();
    }

    private void infoDialog(Stage primaryStage, String message) {
        final Stage dialog = new Stage();
        dialog.initModality(Modality.APPLICATION_MODAL);
        dialog.initOwner(primaryStage);

        Text text = new Text(message);
        text.setFill(Color.BLUE);
        text.setFont(Font.font("Times New Roman", 20));

        ImageView image = new ImageView(new Image("file:info.jpg"));

        Button dialogOk = new Button("OK");
        dialogOk.setPrefWidth(100);

        HBox dialogHbox = new HBox(10, image, text);
        HBox.setMargin(image, new Insets(0, 0, 0, 10));

        VBox dialogVBox = new VBox(10, dialogHbox, dialogOk);
        VBox.setMargin(dialogOk, new Insets(0, 0, 0, 170));

        dialogOk.setOnAction(e -> dialog.close());

        Scene scene = new Scene(dialogVBox, 450, 100);

        dialog.setScene(scene);
        dialog.setTitle("Info Message");
        dialog.show();
    }
}
