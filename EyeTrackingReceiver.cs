using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class EyeTrackingReceiver : MonoBehaviour
{
    private UdpClient udpClient;
    private int port = 5053;
    
    void Start()
    {
        try
        {
            udpClient = new UdpClient(port);
            udpClient.BeginReceive(new System.AsyncCallback(OnDataReceived), null);
            Debug.Log("UDP listening has started. Port: " + port);
        }
        catch (System.Exception e)
        {
            Debug.LogError("UDP beginning error: " + e.Message);
        }
    }
    
    private void OnDataReceived(System.IAsyncResult ar)
    {
        try
        {
            IPEndPoint endPoint = new IPEndPoint(IPAddress.Any, port);
            byte[] receivedBytes = udpClient.EndReceive(ar, ref endPoint);
            
            string receivedData = Encoding.UTF8.GetString(receivedBytes);

            Debug.Log("Received data: " + receivedData);
            
            udpClient.BeginReceive(new System.AsyncCallback(OnDataReceived), null);
        }
        catch (System.Exception e)
        {
            Debug.LogError("UDP receiving error: " + e.Message);
            
            if (udpClient != null)
            {
                udpClient.BeginReceive(new System.AsyncCallback(OnDataReceived), null);
            }
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
