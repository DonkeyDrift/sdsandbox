using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;
using System.Net;
using System.Net.Sockets;

namespace tk
{
    public class TcpClient : MonoBehaviour
    {
        public bool debug = false;

        private Socket _clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        private byte[] _recieveBuffer = new byte[8142];
        private TcpServer _server = null;

        public delegate void OnDataRecv(byte[] data);

        public OnDataRecv onDataRecvCB;

        public delegate void OnConnected();

        public OnConnected onConnectedCB;

        // Flag to let us know a connection has dropped.
        private bool dropped = false;
        public float time_check_dropped = 0.0f;
        public float time_check_dropped_freq = 3.0f;

        /// <summary>
        /// Connect will establish a new TCP socket connection to a remote ip, port. This method is the first method
        /// called to start using this object.
        /// </summary>
        /// <param name="ip"></param>
        /// <param name="port"></param>
        /// <returns></returns>
        public bool Connect(string ip, int port)
        {
            if (_clientSocket.Connected)
                return false;

            try
            {
                IPAddress address = IPAddress.Parse(ip);
                _clientSocket.Connect(new IPEndPoint(address, port));
            }
            catch (SocketException ex)
            {
                Debug.Log(ex.Message);
                return false;
            }

            dropped = false;
            _clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);

            return true;
        }

        /// <summary>
        /// OnServerAccept is an alternate form of client initialization that occurs when a server has accepted a client
        /// connection already and passes that socket in clientSock in a connected state. This also then passes a pointer
        /// to the server which can be used by the clients to broadcast messages to all the peers.
        /// </summary>
        /// <param name="clientSock"></param>
        /// <param name="server"></param>
        /// <returns></returns>
        public bool OnServerAccept(Socket clientSock, TcpServer server)
        {
            if (!clientSock.Connected)
                return false;

            _clientSocket = clientSock;
            _server = server;
            dropped = false;

            _clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);

            return true;
        }

        public void ClientFinishedConnect()
        {
            if(onConnectedCB != null)
                onConnectedCB.Invoke();
        }

        public void ReleaseServer()
        {
            _server = null;
        }

        public void Disconnect()
        {
            try
            {
                if (_clientSocket.Connected)
                {
                    _clientSocket.Shutdown(SocketShutdown.Both);

                    if (!IsDropped())
                    {
                        _clientSocket.Disconnect(true);
                    }
                }
            }
            catch(SocketException e)
            {
                Debug.Log(e.ToString());
            }
            finally
            {
                _clientSocket.Close();
            }

            if (_server != null)
                _server.RemoveClient(this);
        }

        void OnDestroy()
        {
            Disconnect();
        }

        public bool IsDropped()
        {
            return dropped;
        }

        public void Update()
        {
            // Update our drop detection...
            if(_clientSocket != null && !IsDropped())
            {
                time_check_dropped += Time.deltaTime;

                if(!_clientSocket.Connected)
                {
                    dropped = true;
                }
                else if(time_check_dropped > time_check_dropped_freq)
                {
                    time_check_dropped = 0.0f;

                    try
                    {
                        // this is the minimal form of message for a JsonTCPClient
                        string msg = "{}\n";
                        System.Text.Encoding encoding = System.Text.Encoding.Default;
                        _clientSocket.Send(encoding.GetBytes(msg));
                    }
                    catch(SocketException e)
                    {
                        Debug.LogWarning("connection dropped.");
                        dropped = true;
                    }
                }
            }
        }

        private void ReceiveCallback(IAsyncResult AR)
        {
            //Check how much bytes are recieved and call EndRecieve to finalize handshake
            int recieved = 0;

            try
            {
                recieved = _clientSocket.EndReceive(AR);
            }
            catch(SocketException e)
            {
                recieved = 0;
                dropped = true;
                Debug.LogWarning("Exception on recv. Connection dropped.");
            }
            

            if (recieved <= 0)
                return;

            //Copy the recieved data into new buffer , to avoid null bytes
            byte[] recData = new byte[recieved];
            Buffer.BlockCopy(_recieveBuffer, 0, recData, 0, recieved);

            if (debug)
            {
                Debug.Log("recv:" + System.Text.Encoding.Default.GetString(recData));
            }

            //Process data here the way you want , all your bytes will be stored in recData
            if (onDataRecvCB != null)
                onDataRecvCB.Invoke(recData);

            // Reset our drop connection test timer, since we just got data.
            time_check_dropped = 0.0f;

            //Start receiving again
            _clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);
        }

        public bool SendData(byte[] data)
        {
            if (!_clientSocket.Connected)
                return false;

            SocketAsyncEventArgs socketAsyncData = new SocketAsyncEventArgs();
            socketAsyncData.SetBuffer(data, 0, data.Length);
            _clientSocket.SendAsync(socketAsyncData);

            if (debug)
            {
                Debug.Log("sent:" + System.Text.Encoding.Default.GetString(data));
            }

            return true;
        }

        // ---- 池化大报文发送(遥测专用,零分配) ----
        // 遥测报文 ~200KB@60Hz,旧路径每帧新建 byte[] 产生 ~12MB/s GC 垃圾,
        // 非增量 GC 全停主线程 → 窗口不定时卡顿闪动。这里维护固定大缓冲池:
        // RentSendBuffer 租缓冲 → 调用方直接写包 → CommitPooledSend 排队。
        // 发送串行化(任意时刻至多一个 BeginSend 在飞,其余 FIFO 排队,
        // 保证 TCP 字节序与报文顺序);完成回调归还缓冲并抽下一个。
        // 池耗尽时租用失败由调用方丢帧:遥测是实时数据,丢帧优于积压旧帧。
        public int pooledSendDrops { get; private set; }

        const int POOLED_BUF_COUNT = 8;
        const int POOLED_BUF_SIZE = 1 << 20; // 1MB:640x480 JPEG base64 后 ~200KB,双相机也够
        byte[][] _pooledBufs;
        bool[] _pooledBusy;
        bool _pooledSendInFlight;
        readonly Queue<KeyValuePair<byte[], int>> _pooledSendQueue = new Queue<KeyValuePair<byte[], int>>();
        readonly object _pooledLock = new object();

        void EnsurePooledSendLocked()
        {
            if (_pooledBufs != null)
                return;
            _pooledBufs = new byte[POOLED_BUF_COUNT][];
            _pooledBusy = new bool[POOLED_BUF_COUNT];
            for (int i = 0; i < POOLED_BUF_COUNT; i++)
                _pooledBufs[i] = new byte[POOLED_BUF_SIZE];
        }

        /// <summary>租一块发送缓冲;池耗尽返回 null(调用方应丢帧并计数)。</summary>
        public byte[] RentSendBuffer()
        {
            lock (_pooledLock)
            {
                EnsurePooledSendLocked();
                for (int i = 0; i < POOLED_BUF_COUNT; i++)
                {
                    if (!_pooledBusy[i])
                    {
                        _pooledBusy[i] = true;
                        return _pooledBufs[i];
                    }
                }
                pooledSendDrops++;
                return null;
            }
        }

        /// <summary>归还未提交发送的缓冲(拼包中途放弃时用)。</summary>
        public void ReturnSendBuffer(byte[] buf)
        {
            if (buf == null)
                return;
            lock (_pooledLock) { ReturnPooledLocked(buf); }
        }

        void ReturnPooledLocked(byte[] buf)
        {
            for (int i = 0; i < POOLED_BUF_COUNT; i++)
            {
                if (ReferenceEquals(_pooledBufs[i], buf))
                {
                    _pooledBusy[i] = false;
                    return;
                }
            }
        }

        /// <summary>提交已写好的缓冲发送;失败(未连接/已断开)归还缓冲并返回 false。</summary>
        public bool CommitPooledSend(byte[] buf, int length)
        {
            if (buf == null || length <= 0)
            {
                ReturnSendBuffer(buf);
                return false;
            }
            lock (_pooledLock)
            {
                if (dropped || !_clientSocket.Connected)
                {
                    ReturnPooledLocked(buf);
                    pooledSendDrops++;
                    return false;
                }
                _pooledSendQueue.Enqueue(new KeyValuePair<byte[], int>(buf, length));
                PumpPooledSendLocked();
            }
            return true;
        }

        void PumpPooledSendLocked()
        {
            while (!_pooledSendInFlight && _pooledSendQueue.Count > 0)
            {
                if (dropped || !_clientSocket.Connected)
                {
                    while (_pooledSendQueue.Count > 0)
                        ReturnPooledLocked(_pooledSendQueue.Dequeue().Key);
                    return;
                }
                KeyValuePair<byte[], int> item = _pooledSendQueue.Dequeue();
                _pooledSendInFlight = true;
                try
                {
                    _clientSocket.BeginSend(item.Key, 0, item.Value, SocketFlags.None, PooledSendCallback, item.Key);
                }
                catch (Exception)
                {
                    _pooledSendInFlight = false;
                    ReturnPooledLocked(item.Key);
                    dropped = true;
                }
            }
        }

        void PooledSendCallback(IAsyncResult ar)
        {
            byte[] buf = (byte[])ar.AsyncState;
            try
            {
                _clientSocket.EndSend(ar);
            }
            catch (Exception)
            {
                dropped = true;
            }
            lock (_pooledLock)
            {
                _pooledSendInFlight = false;
                ReturnPooledLocked(buf);
                PumpPooledSendLocked();
            }
        }

        public bool SendDataToPeers(byte[] data)
        {
            if (!_clientSocket.Connected || _server == null)
                return false;

            _server.SendData(data, this);
            return true;
        }

        public void SetDebug(bool _debug)
        {
            debug = _debug;
        }

    }

} //end namepace tk
