using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class EyeTrackingReceiver : MonoBehaviour
{
    [Header("UI References")]
    public RectTransform selectionHighlight;
    public GameObject confirmationText;
    
    [Header("Grid Settings")]
    public float buttonWidth = 400f;
    public float buttonHeight = 250f;
    public float buttonSpacing = 30f;
    
    [Header("Movement Settings")]
    public float movementCooldown = 0.8f;
    public float blinkThreshold = 1.2f;  
    
    [Header("UDP Settings")]
    private int port = 5053;
    
    // Grid pozisyon sistemi
    private int currentRow = 2; 
    private int currentCol = 0;
    

    private Vector2[,] gridPositions = new Vector2[3, 3];
    

    private UdpClient udpClient;
    private string lastReceivedMessage = "Waiting for data...";
    private string currentGazeDirection = "CENTER";
    private bool isCurrentlyBlinking = false;
    

    private float lastMoveTime = 0f;
    private bool isBlinking = false;
    private float blinkStartTime = 0f;
    
    void Start()
    {
        InitializeGrid();
        InitializeUDP();
        UpdateHighlightPosition();
        
        if (confirmationText != null)
            confirmationText.SetActive(false);
    }
    
    void InitializeGrid()
    {

        float startX = -430f; 
        float startY = 280f;
        
        float stepX = buttonWidth + buttonSpacing;
        float stepY = buttonHeight + buttonSpacing;
        
        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                float posX = startX + (col * stepX);
                float posY = startY - (row * stepY); 
                gridPositions[row, col] = new Vector2(posX, posY);
            }
        }
        
        Debug.Log("Grid initialized (Fixed Y-axis):");
        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                Debug.Log($"Grid[{row},{col}] = {gridPositions[row, col]}");
            }
        }
    }
    
    void InitializeUDP()
    {
        try
        {
            udpClient = new UdpClient(port);
            udpClient.BeginReceive(new System.AsyncCallback(OnDataReceived), null);
            Debug.Log($"UDP dinleme başladı. Port: {port}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"UDP başlatma hatası: {e.Message}");
        }
    }
    
    private void OnDataReceived(System.IAsyncResult ar)
    {
        try
        {
            IPEndPoint endPoint = new IPEndPoint(IPAddress.Any, port);
            byte[] receivedBytes = udpClient.EndReceive(ar, ref endPoint);
            string receivedData = Encoding.UTF8.GetString(receivedBytes);
            
            lastReceivedMessage = receivedData;
            
            // Gaze direction ve blink durumu parse et
            if (receivedData.Contains("GAZE:") && receivedData.Contains("BLINK:"))
            {
                // Format: "GAZE:LEFT,BLINK:true,EAR:0.25"
                string[] parts = receivedData.Split(',');
                
                foreach (string part in parts)
                {
                    if (part.StartsWith("GAZE:"))
                    {
                        currentGazeDirection = part.Substring(5);
                    }
                    else if (part.StartsWith("BLINK:"))
                    {
                        string blinkStr = part.Substring(6);
                        isCurrentlyBlinking = blinkStr == "True" || blinkStr == "true";
                    }
                }
            }
            else
            {
                // Sadece gaze direction
                currentGazeDirection = receivedData.Trim();
                isCurrentlyBlinking = false;
            }
            
            // Asenkron dinleme işlemine devam et
            udpClient.BeginReceive(new System.AsyncCallback(OnDataReceived), null);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"UDP alma hatası: {e.Message}");
            
            if (udpClient != null)
            {
                udpClient.BeginReceive(new System.AsyncCallback(OnDataReceived), null);
            }
        }
    }
    
    void Update()
    {
        ProcessGazeMovement();
        ProcessBlinking();
    }
    
    void ProcessGazeMovement()
    {
        // Hareket cooldown kontrolü
        if (Time.time - lastMoveTime < movementCooldown)
            return;
            
        int newRow = currentRow;
        int newCol = currentCol;
        bool moved = false;
        
        switch (currentGazeDirection)
        {
            case "LEFT":
                if (currentCol > 0)
                {
                    newCol = currentCol - 1;
                    moved = true;
                }
                break;
                
            case "RIGHT":
                if (currentCol < 2)
                {
                    newCol = currentCol + 1;
                    moved = true;
                }
                break;
                
            case "UP":
                if (currentRow > 0)
                {
                    newRow = currentRow - 1;
                    moved = true;
                }
                break;
                
            case "DOWN":
                if (currentRow < 2)
                {
                    newRow = currentRow + 1;
                    moved = true;
                }
                break;
        }
        
        if (moved)
        {
            currentRow = newRow;
            currentCol = newCol;
            UpdateHighlightPosition();
            lastMoveTime = Time.time;
            
            Debug.Log($"Moved to Grid[{currentRow},{currentCol}] = {gridPositions[currentRow, currentCol]}");
        }
    }
    
    void ProcessBlinking()
    {
        if (isCurrentlyBlinking && !isBlinking)
        {
            isBlinking = true;
            blinkStartTime = Time.time;
            Debug.Log("Blink started");
        }
        else if (!isCurrentlyBlinking && isBlinking)
        {
            float blinkDuration = Time.time - blinkStartTime;
            Debug.Log($"Blink ended. Duration: {blinkDuration:F2}s");
            
            if (blinkDuration >= blinkThreshold)
            {
                ConfirmSelection();
            }
            
            isBlinking = false;
        }
    }
    
    void UpdateHighlightPosition()
    {
        if (selectionHighlight != null)
        {
            Vector2 targetPosition = gridPositions[currentRow, currentCol];
            selectionHighlight.anchoredPosition = targetPosition;
            
            Debug.Log($"Highlight position updated to: {targetPosition}");
        }
    }
    
    void ConfirmSelection()
    {
        if (currentRow == 0 && currentCol == 2)
        {
            currentRow = 2;
            currentCol = 0;
            UpdateHighlightPosition();
            
            if (confirmationText != null)
            {
                confirmationText.SetActive(true);
                Invoke("HideConfirmation", 1.5f);
            }
            
            Debug.Log("CANCELLED! Returned to starting position Grid[2,0]");
        }
        else
        {
            if (confirmationText != null)
            {
                confirmationText.SetActive(true);
                Invoke("HideConfirmation", 2f);
            }
            
            Debug.Log($"Selection CONFIRMED at Grid[{currentRow},{currentCol}]");
        }
    }
    
    void HideConfirmation()
    {
        if (confirmationText != null)
        {
            confirmationText.SetActive(false);
        }
    }
    
    void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 400, 30), "Gaze Direction: " + currentGazeDirection);
        GUI.Label(new Rect(10, 40, 400, 30), $"Current Position: Grid[{currentRow},{currentCol}]");
        GUI.Label(new Rect(10, 70, 400, 30), $"Pixel Position: {gridPositions[currentRow, currentCol]}");
        GUI.Label(new Rect(10, 100, 400, 30), $"UDP Status: Listening on port {port}");
        
        if (currentRow == 0 && currentCol == 2)
        {
            GUI.Label(new Rect(10, 130, 400, 30), ">>> CANCEL BUTTON SELECTED <<<");
        }
        
        if (isBlinking)
        {
            float blinkDuration = Time.time - blinkStartTime;
            GUI.Label(new Rect(10, 160, 400, 30), $"BLINKING: {blinkDuration:F1}s / {blinkThreshold:F1}s");
            
            if (blinkDuration >= blinkThreshold)
            {
                if (currentRow == 0 && currentCol == 2)
                {
                    GUI.Label(new Rect(10, 190, 400, 30), "READY TO CANCEL!");
                }
                else
                {
                    GUI.Label(new Rect(10, 190, 400, 30), "READY TO CONFIRM!");
                }
            }
        }
        
        float timeSinceLastMove = Time.time - lastMoveTime;
        if (timeSinceLastMove < movementCooldown)
        {
            float remaining = movementCooldown - timeSinceLastMove;
            GUI.Label(new Rect(10, 220, 400, 30), $"Movement cooldown: {remaining:F1}s");
        }
    }
    
    void OnApplicationQuit()
    {
        if (udpClient != null)
        {
            udpClient.Close();
        }
    }
    
    void OnDestroy()
    {
        OnApplicationQuit();
    }
}
